class DischargePlot:
    """
    Minimal deterministische Darstellung der Entladung in zwei Subplots.

    Regeln:
      - X default = results["relevant_time_window"] (an Datenzeit geklemmt).
      - Y = min/max der sichtbaren Daten im aktuellen X-Fenster (ohne Padding).
        * oben: nur Originalsignal U(t)
        * unten: NUR raw dU/dt (MA wird NICHT fürs Scaling berücksichtigt)

    Public API:
      - draw(initial_view="window") -> Figure
      - set_xlim(x_left, x_right)
      - get_xlim() -> (x_left, x_right)
      - zoom_window()
      - zoom_full()
      - pan(delta_s)
      - zoom(factor, center=None)     # factor < 1: rein; > 1: raus
      - update_results(results, keep_view=True)
      - get_view() -> {"xlim": (...), "ylim_top": (...), "ylim_bot": (...)}
      - get_axes() -> (ax_top, ax_bot)
    """

    # -------------------------- Init --------------------------

    def __init__(self, signal, results: dict):
        self.signal = signal
        self.results = results
        self.params = results.get("params_used", {}) or {}
        self.dt = float(self.params.get("sampling_interval", 0.01))

        self.fig = None
        self.ax_top = None
        self.ax_bot = None

        # X-Ansicht (einzige Quelle der Wahrheit)
        self.current_xlim = None

        # Optional-Flags (falls du später togglen willst)
        self.show_peak_line = True
        self.show_u3 = True
        self.show_derivative_raw = True
        self.show_derivative_ma = True

        # Cache für Nulldurchgang des Fits
        self._t_zero = None

    # -------------------------- Public ------------------------

    def draw(self, initial_view: str = "window"):
        """
        initial_view: "window" (relevant_time_window) oder "full" (bis Fit==0 ab rated_time)
        """
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from cap_signal_processing import evaluate_polynomial

        r = self.results
        s = self.signal

        # Ableitung vorhanden?
        if s.data["derivative"].isnull().all():
            s.get_derivative()

        peak_time = float(r["peak_time"])
        rated_t = float(r["rated_time"])
        U3 = float(r["U3"])
        coeffs = r["post_peak_unloading_fit"]

        # X-Initialisierung
        self.current_xlim = self._initial_xlim(initial_view)

        # Fit-Segment (nur für Darstellung, NICHT fürs Y-Scaling)
        t_zero = self._compute_t_zero()
        t_fit_seg, y_fit_seg = self._fit_segment(peak_time, t_zero, coeffs)

        # U3-Niveaus (nur Darstellung)
        y_fit_at_peak = float(evaluate_polynomial(coeffs, peak_time))
        y_fit_plus_u3 = y_fit_at_peak + U3

        # Daten
        import numpy as np
        t_data = np.asarray(s.data["time"], dtype=float)
        v_data = np.asarray(s.data["value"], dtype=float)

        # Figure & Axes
        self.fig = Figure(figsize=(9.6, 6.6), dpi=100)
        self.ax_top = self.fig.add_subplot(211)
        self.ax_bot = self.fig.add_subplot(212, sharex=self.ax_top)

        # --------- oben: U(t) ----------
        self.ax_top.set_xlim(self.current_xlim)

        # Plot Daten
        self.ax_top.plot(t_data, v_data, label="Signal", linewidth=1.5)

        # Fit dünn (Peak -> t_zero)
        if len(t_fit_seg) > 1:
            self.ax_top.plot(t_fit_seg, y_fit_seg, label="Unloading-Fit", linewidth=1.0)

        # U3-Linien (ohne Einfluss aufs Scaling)
        if self.show_u3:
            self.ax_top.axhline(y_fit_at_peak, linestyle="--", linewidth=1.0)
            self.ax_top.axhline(y_fit_plus_u3, linestyle="--", linewidth=1.0)
            # Dummy-Handle für Legende
            dummy_u3 = Line2D([0], [0], linestyle="--", color=self.ax_top.lines[-1].get_color(), label="U3")
        else:
            dummy_u3 = None

        # Peak-Vertikale (ohne Einfluss aufs Scaling)
        if self.show_peak_line:
            self.ax_top.axvline(peak_time, linestyle="--", linewidth=1.0)

        self.ax_top.set_ylabel("U (V)")
        self.ax_top.grid(True, alpha=0.35)

        # Legende kompakt
        handles, labels = self.ax_top.get_legend_handles_labels()
        if self.show_u3 and dummy_u3 is not None:
            handles.append(dummy_u3)
            labels.append("U3")
        if handles:
            self.ax_top.legend(handles, labels, loc="upper right")

        # --------- unten: dU/dt ----------
        d_times = t_data
        d_raw = np.asarray(s.data["derivative"], dtype=float)
        if self.show_derivative_raw:
            self.ax_bot.plot(d_times, d_raw, label="dU/dt", linewidth=1.2)

        sm = r.get("smoothed_derivative_signal", None)
        if self.show_derivative_ma and sm is not None:
            n = self.params.get("derivative_smooth_n")
            label = "MA(dU/dt)" + (f" [n={n}]" if n is not None else "")
            self.ax_bot.plot(sm.data["time"], sm.data["value"], label=label, linewidth=0.9)

        self.ax_bot.set_xlabel("Zeit (s)")
        self.ax_bot.set_ylabel("dU/dt (V/s)")
        self.ax_bot.grid(True, alpha=0.25)
        self.ax_bot.legend(loc="upper left")

        # Y-Skalierung NUR nach sichtbaren Daten (oben: U; unten: raw dU/dt)
        self._autoscale_y_all()

        # Titel & Layout
        self.fig.suptitle(f"{r.get('file_name', 'Signal')} – Discharge & Fit (U3)")
        self.fig.tight_layout()
        return self.fig

    def set_xlim(self, x_left: float, x_right: float):
        """Setzt die aktuelle Ansicht und wendet sie an."""
        self.current_xlim = (float(x_left), float(x_right))
        self._apply_view()

    def get_xlim(self):
        return tuple(self.current_xlim) if self.current_xlim is not None else None

    def zoom_window(self):
        """Zurück auf relevant_time_window (an Datenzeit geklemmt)."""
        self.current_xlim = self._initial_xlim("window")
        self._apply_view()

    def zoom_full(self):
        """Von max(t0, rated_time) bis Nulldurchgang des Fits."""
        t0, t1 = self._full_xlim()
        self.current_xlim = (t0, t1)
        self._apply_view()

    def pan(self, delta_s: float):
        """X-Fenster um delta_s verschieben (an Datenzeit klemmen)."""
        t0, t1 = self._data_time_bounds()
        a, b = self.current_xlim
        w = b - a
        a_new = max(t0, a + delta_s)
        b_new = a_new + w
        if b_new > t1:
            b_new = t1
            a_new = b_new - w
        self.current_xlim = (a_new, b_new)
        self._apply_view()

    def zoom(self, factor: float, center: float | None = None):
        """Zoom um Faktor (factor<1 rein, >1 raus). center in s; sonst Fenster-Mitte."""
        a, b = self.current_xlim
        if center is None:
            center = 0.5 * (a + b)
        new_half = 0.5 * (b - a) * float(factor)
        a_new = center - new_half
        b_new = center + new_half
        # An Datenzeit klemmen
        t0, t1 = self._data_time_bounds()
        if a_new < t0:
            shift = t0 - a_new
            a_new, b_new = a_new + shift, b_new + shift
        if b_new > t1:
            shift = b_new - t1
            a_new, b_new = a_new - shift, b_new - shift
        # Mindestbreite vermeiden Kollaps
        if b_new <= a_new:
            eps = 1e-9
            b_new = a_new + eps
        self.current_xlim = (a_new, b_new)
        self._apply_view()

    def update_results(self, results: dict, keep_view: bool = True):
        """Ersetzt results; behält View optional bei."""
        old_xlim = tuple(self.current_xlim) if (keep_view and self.current_xlim) else None
        self.results = results
        self.params = results.get("params_used", {}) or {}
        self.dt = float(self.params.get("sampling_interval", 0.01))
        self._t_zero = None
        # neu zeichnen
        self.draw(initial_view="window" if not keep_view else "window")
        if old_xlim:
            self.current_xlim = old_xlim
            self._apply_view()

    def get_view(self) -> dict:
        return {
            "xlim": tuple(self.ax_top.get_xlim()),
            "ylim_top": tuple(self.ax_top.get_ylim()),
            "ylim_bot": tuple(self.ax_bot.get_ylim()),
        }

    def get_axes(self):
        return self.ax_top, self.ax_bot

    # ------------------------ Internals ------------------------

    def _initial_xlim(self, initial_view: str):
        """Bestimmt das Startfenster. 'window' = genau relevant_time_window; 'full' = rated_time..t_zero."""
        if initial_view == "full":
            return self._full_xlim()
        # window
        rel = self.results.get("relevant_time_window")
        if rel is None:
            rated_t = float(self.results["rated_time"])
            rel = (rated_t, rated_t + 10.0)
        a, b = float(rel[0]), float(rel[1])
        t0, t1 = self._data_time_bounds()
        return (max(t0, a), min(t1, b))

    def _full_xlim(self):
        """Von max(t0, rated_time) bis zum Nulldurchgang des Fits."""
        t0_data, t1_data = self._data_time_bounds()
        rated_t = float(self.results["rated_time"])
        a = max(t0_data, rated_t)
        t_zero = self._compute_t_zero()
        b = min(t1_data, float(t_zero))
        if b <= a:
            b = min(t1_data, a + max(self.dt, 1e-6))
        return (a, b)

    def _apply_view(self):
        """Wendet self.current_xlim an und skaliert Y danach strikt auf sichtbare Daten."""
        if self.ax_top is None or self.ax_bot is None:
            return
        self.ax_top.set_xlim(self.current_xlim)
        self._autoscale_y_all()

    def _autoscale_y_all(self):
        """Skaliert Y beider Subplots strikt nach min/max der sichtbaren Daten."""
        import numpy as np
        # --- oben: nur Originalsignal ---
        t = np.asarray(self.signal.data["time"], dtype=float)
        y = np.asarray(self.signal.data["value"], dtype=float)
        a, b = self.current_xlim
        mask = (t >= a) & (t <= b)
        if np.any(mask):
            ymin, ymax = float(np.min(y[mask])), float(np.max(y[mask]))
        else:
            ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymin == ymax:
            eps = 1e-12
            ymin -= eps
            ymax += eps
        self.ax_top.set_ylim(ymin, ymax)

        # --- unten: NUR raw dU/dt fürs Scaling (hier raw verwenden) ---
        d_raw = np.asarray(self.signal.data["derivative"], dtype=float)
        if np.any(mask):
            ymin, ymax = float(np.min(d_raw[mask])), float(np.max(d_raw[mask]))
        else:
            ymin, ymax = float(np.min(d_raw)), float(np.max(d_raw))
        if ymin == ymax:
            eps = 1e-12
            ymin -= eps
            ymax += eps
        self.ax_bot.set_ylim(ymin, ymax)

    def _compute_t_zero(self) -> float:
        """Nulldurchgang des Unloading-Fits ab peak_time bis Datenende (lineare Interpolation)."""
        import numpy as np
        from cap_signal_processing import evaluate_polynomial

        if self._t_zero is not None:
            return self._t_zero

        coeffs = self.results["post_peak_unloading_fit"]
        peak_time = float(self.results["peak_time"])
        t = np.asarray(self.signal.data["time"], dtype=float)
        t_end = float(t[-1])

        t_fit = np.arange(peak_time, t_end + 0.5*self.dt, self.dt)
        y_fit = np.array([evaluate_polynomial(coeffs, _t) for _t in t_fit], dtype=float)

        idx = np.where(y_fit <= 0.0)[0]
        if idx.size:
            k = int(idx[0])
            if k == 0:
                t_zero = t_fit[0]
            else:
                t1, y1 = t_fit[k-1], y_fit[k-1]
                t2, y2 = t_fit[k],   y_fit[k]
                t_zero = t1 + (0.0 - y1) * (t2 - t1) / (y2 - y1) if (y2 - y1) != 0 else t2
        else:
            t_zero = t_end

        self._t_zero = float(t_zero)
        return self._t_zero

    def _fit_segment(self, peak_time: float, t_zero: float, coeffs):
        """Zeit-/Wertvektoren für dünne Fit-Darstellung (Peak -> t_zero)."""
        import numpy as np
        from cap_signal_processing import evaluate_polynomial
        if t_zero <= peak_time:
            y = float(evaluate_polynomial(coeffs, peak_time))
            return np.array([peak_time, peak_time]), np.array([y, y])
        t = np.arange(peak_time, t_zero + 0.5*self.dt, self.dt)
        y = np.array([evaluate_polynomial(coeffs, _t) for _t in t], dtype=float)
        return t, y

    def _data_time_bounds(self):
        """(t_min, t_max) aus den Messdaten."""
        import numpy as np
        t = np.asarray(self.signal.data["time"], dtype=float)
        return float(t[0]), float(t[-1])
