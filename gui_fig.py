class DischargePlot:
    """
    Kapselt die Discharge-/U3-Darstellung in zwei Subplots.

    Public API:
      - draw(view="window", xlim=None, ...)
          -> Figure
      - set_xlim(x_left, x_right)
      - zoom_window()
      - zoom_full()
      - update_results(results)
      - get_view() -> {"xlim": (..,..), "ylim_top": (..,..), "ylim_bot": (..,..)}
      - get_axes() -> (ax_top, ax_bot)
    """
    def __init__(self, signal, results: dict):
        self.signal = signal
        self.results = results
        self.params = results.get("params_used", {}) or {}
        self.dt = float(self.params.get("sampling_interval", 0.01))

        self.fig = None
        self.ax_top = None
        self.ax_bot = None
        self._t_zero = None  # Cache: Nulldurchgang des Fits

    # ---------- Public ----------

    def draw(
        self,
        view: str = "window",
        xlim: tuple[float, float] | None = None,
        show_peak_line: bool = True,
        show_u3: bool = True,
        show_derivative_raw: bool = True,
        show_derivative_ma: bool = True,
    ):
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from cap_signal_processing import evaluate_polynomial

        r = self.results
        s = self.signal

        peak_time = float(r["peak_time"])
        U3        = float(r["U3"])
        coeffs    = r["post_peak_unloading_fit"]
        rated_t   = float(r["rated_time"])
        sm_der    = r.get("smoothed_derivative_signal", None)

        if s.data["derivative"].isnull().all():
            s.get_derivative()

        # discharge end
        t_zero = self._compute_t_zero()

        t_data = np.asarray(s.data["time"], dtype=float)
        v_data = np.asarray(s.data["value"], dtype=float)

        # X limits
        x_left, x_right = self._choose_xlim(view, xlim, rated_t, t_zero, t_data)

        # Fit segment
        t_fit_seg, y_fit_seg = self._fit_segment(peak_time, t_zero, coeffs)

        # U3 levels
        y_fit_at_peak = float(evaluate_polynomial(coeffs, peak_time))
        y_fit_plus_u3 = y_fit_at_peak + U3

        # Figure/Subplots
        self.fig = Figure(figsize=(9.6, 6.6), dpi=100)
        self.ax_top = self.fig.add_subplot(211)
        self.ax_bot = self.fig.add_subplot(212, sharex=self.ax_top)

        # --- Top plot: U(t) ---
        ax = self.ax_top
        ax.set_xlim((x_left, x_right))

        vmin_vmax = self._autoscale_voltage_ylim(x_left, x_right, t_fit_seg, y_fit_seg, y_fit_at_peak, y_fit_plus_u3)
        if vmin_vmax is not None:
            vmin, vmax = vmin_vmax
            ax.set_ylim(vmin, vmax)

        ax.plot(t_data, v_data, label="Signal", linewidth=1.5)
        if len(t_fit_seg) > 1:
            ax.plot(t_fit_seg, y_fit_seg, label="Unloading-Fit", linewidth=1.0)

        if show_u3:
            ax.axhline(y_fit_at_peak, linestyle="--", linewidth=1.0)
            ax.axhline(y_fit_plus_u3, linestyle="--", linewidth=1.0)
            dummy_u3 = Line2D([0], [0], linestyle="--", color=ax.lines[-1].get_color(), label="U3")
        else:
            dummy_u3 = None

        if show_peak_line:
            ax.axvline(peak_time, linestyle="--", linewidth=1.0)

        ax.set_ylabel("U (V)")
        ax.grid(True, alpha=0.35)

        handles, labels = ax.get_legend_handles_labels()
        if show_u3 and dummy_u3 is not None:
            handles.append(dummy_u3)
            labels.append("U3")
        ax.legend(handles, labels, loc="upper right")

        # --- Bottom plot: dU/dt ---
        ax = self.ax_bot
        d_times = t_data
        d_raw   = np.asarray(s.data["derivative"], dtype=float)

        if show_derivative_raw:
            ax.plot(d_times, d_raw, label="dU/dt", linewidth=1.2)
        if show_derivative_ma and sm_der is not None:
            n = self.params.get("derivative_smooth_n")
            label = "MA(dU/dt)" + (f" [n={n}]" if n is not None else "")
            ax.plot(sm_der.data["time"], sm_der.data["value"], label=label, linewidth=0.9)

        ax.set_xlabel("Zeit (s)")
        ax.set_ylabel("dU/dt (V/s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

        self._autoscale_derivative_ylim(x_left, x_right)

        self.fig.suptitle(f"{r.get('file_name', 'Signal')} – Discharge & Fit (U3)")
        self.fig.tight_layout()
        return self.fig

    def set_xlim(self, x_left: float, x_right: float):
        self.ax_top.set_xlim((x_left, x_right))
        self._autoscale_derivative_ylim(x_left, x_right)

    def zoom_window(self):
        r = self.results
        rated_t = float(r["rated_time"])
        rel = r.get("relevant_time_window", (rated_t, rated_t + 10.0))
        self.set_xlim(float(rel[0]), float(rel[1]))

    def zoom_full(self):
        import numpy as np
        r = self.results
        rated_t = float(r["rated_time"])
        t_zero = self._compute_t_zero()
        t_data = np.asarray(self.signal.data["time"], dtype=float)
        x_left = max(float(t_data[0]), rated_t)
        x_right = max(t_zero, rated_t + 1.0)
        self.set_xlim(x_left, x_right)

    def update_results(self, results: dict):
        cur = self.get_view()
        self.results = results
        self.params = results.get("params_used", {}) or {}
        self.dt = float(self.params.get("sampling_interval", 0.01))
        self._t_zero = None

        self.draw(view="window")
        self.set_xlim(*cur["xlim"])

    def get_view(self) -> dict:
        xlim = tuple(self.ax_top.get_xlim())
        ylim_top = tuple(self.ax_top.get_ylim())
        ylim_bot = tuple(self.ax_bot.get_ylim())
        return {"xlim": xlim, "ylim_top": ylim_top, "ylim_bot": ylim_bot}

    def get_axes(self):
        return self.ax_top, self.ax_bot

    # ---------- Internal ----------

    def _compute_t_zero(self) -> float:
        import numpy as np
        from cap_signal_processing import evaluate_polynomial

        if self._t_zero is not None:
            return self._t_zero

        r = self.results
        s = self.signal
        coeffs = r["post_peak_unloading_fit"]
        peak_time = float(r["peak_time"])

        t_data = np.asarray(s.data["time"], dtype=float)
        t_end = float(t_data[-1])

        t_fit = np.arange(peak_time, t_end + 0.5*self.dt, self.dt)
        y_fit = np.array([evaluate_polynomial(coeffs, t) for t in t_fit], dtype=float)

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

    def _choose_xlim(self, view, xlim, rated_t, t_zero, t_data):
        """
        X-Limits auswählen:
        - xlim: expliziter Override
        - view="full": von rated_time bis zum (berechneten) Nulldurchgang t_zero
        - view="window" (Default): EXAKT results["relevant_time_window"], nur weich an Datenbereich geklemmt
        """
        r = self.results
        if xlim is not None:
            return float(xlim[0]), float(xlim[1])

        # "full": bis zum Ende des Entladevorgangs
        if view == "full":
            x_left = max(float(t_data[0]), float(rated_t))
            x_right = float(min(float(t_data[-1]), float(t_zero)))
            return x_left, x_right

        # "window" (Default): strictly relevant_time_window (nur clamping an Daten)
        rel = r.get("relevant_time_window", (rated_t, rated_t + 10.0))
        x_left = max(float(t_data[0]), float(rel[0]))
        x_right = min(float(t_data[-1]), float(rel[1]))
        return x_left, x_right


    def _fit_segment(self, peak_time, t_zero, coeffs):
        import numpy as np
        from cap_signal_processing import evaluate_polynomial
        if t_zero <= peak_time:
            return np.array([peak_time, peak_time]), np.array([evaluate_polynomial(coeffs, peak_time)]*2)
        t = np.arange(peak_time, t_zero + 0.5*self.dt, self.dt)
        y = np.array([evaluate_polynomial(coeffs, _t) for _t in t], dtype=float)
        return t, y

    def _autoscale_voltage_ylim(self, x_left, x_right, t_fit_seg, y_fit_seg, y_fit_at_peak, y_fit_plus_u3):
        import numpy as np
        t = np.asarray(self.signal.data["time"], dtype=float)
        y = np.asarray(self.signal.data["value"], dtype=float)
        m = (t >= x_left) & (t <= x_right)
        if np.any(m):
            vmin = float(np.min(y[m]))
            vmax = float(np.max(y[m]))
        else:
            return None
        if t_fit_seg is not None and len(t_fit_seg) > 1:
            vmin = min(vmin, float(np.min(y_fit_seg)))
            vmax = max(vmax, float(np.max(y_fit_seg)))
        vmin = min(vmin, float(min(y_fit_at_peak, y_fit_plus_u3)))
        vmax = max(vmax, float(max(y_fit_at_peak, y_fit_plus_u3)))
        span = vmax - vmin
        pad = 0.02 * (span if span > 0 else 1.0)
        return vmin - pad, vmax + pad

    def _autoscale_derivative_ylim(self, x_left, x_right):
        import numpy as np, math
        t = np.asarray(self.signal.data["time"], dtype=float)
        y_raw = np.asarray(self.signal.data["derivative"], dtype=float)

        y_min, y_max = np.inf, -np.inf
        m = (t >= x_left) & (t <= x_right)
        if np.any(m):
            y_min = min(y_min, float(np.min(y_raw[m])))
            y_max = max(y_max, float(np.max(y_raw[m])))

        sm = self.results.get("smoothed_derivative_signal", None)
        if sm is not None:
            ts = np.asarray(sm.data["time"], dtype=float)
            ys = np.asarray(sm.data["value"], dtype=float)
            ms = (ts >= x_left) & (ts <= x_right)
            if np.any(ms):
                y_min = min(y_min, float(np.min(ys[ms])))
                y_max = max(y_max, float(np.max(ys[ms])))

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            self.ax_bot.set_ylim(-1e-3, 1e-3)
            return

        span = y_max - y_min
        if span <= 0:
            step = self._nice_step(1e-6)
            self.ax_bot.set_ylim(y_min - 2*step, y_max + 2*step)
            return

        step = self._nice_step(span)
        y0 = (math.floor(y_min / step)) * step
        y1 = (math.ceil (y_max / step)) * step
        if y1 == y0:
            y1 = y0 + step
        self.ax_bot.set_ylim(y0, y1)

    def _nice_step(self, span):
        import math
        if span <= 0:
            return 1.0
        exp = math.floor(math.log10(span))
        base = 10 ** exp
        candidates = [1, 2, 5]
        best_m = candidates[0]
        best_score = 1e9
        for m in candidates:
            ticks = span / (m * base)
            score = (ticks - 8.0) ** 2
            if score < best_score:
                best_score = score
                best_m = m
        return best_m * base
