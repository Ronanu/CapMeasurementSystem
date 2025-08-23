def make_figure(signal, results: dict, xlim: tuple[float, float] | None = None, view: str = "window"):
    """
    Zweizeiliger Plot:
      (oben)  U(t) mit dünnem Unloading-Fit von peak_time bis U=0, vertikale Peak-Linie, U3-Linien
      (unten) dU/dt (raw) + MA(dU/dt) (dünn), MA-Parameter im Legendenlabel

    Initialansicht:
      - Standard: `view="window"` → uses results["relevant_time_window"]
      - `view="full"` → von (max(t_data[0], rated_time)) bis zum Nulldurchgang des Fits
      - `xlim=(xmin, xmax)` übersteuert beides (höchste Priorität)

    Parameter:
      xlim : optionales (xmin, xmax)
      view : "window" | "full"
    """
    import numpy as np
    from matplotlib.figure import Figure
    from cap_signal_processing import evaluate_polynomial

    # --- Inputs & Params ---
    peak_time = float(results["peak_time"])
    U3       = float(results["U3"])
    coeffs   = results["post_peak_unloading_fit"]
    rated_t  = float(results["rated_time"])
    params   = results.get("params_used", {})
    dt       = float(params.get("sampling_interval", 0.01))
    sm_der   = results.get("smoothed_derivative_signal", None)

    # Ableitung sicherstellen
    if signal.data["derivative"].isnull().all():
        signal.get_derivative()

    # Fit-Level & U3-Linien
    y_fit_at_peak = float(evaluate_polynomial(coeffs, peak_time))
    y_fit_plus_u3 = y_fit_at_peak + U3

    # Datenzeitachse
    t_data = np.asarray(signal.data["time"], dtype=float)
    v_data = np.asarray(signal.data["value"], dtype=float)
    t0     = float(t_data[0])
    t_end  = float(t_data[-1])

    # Fit-Raster über komplette Entladung
    t_fit_full = np.arange(peak_time, t_end + 0.5*dt, dt)
    y_fit_full = np.array([evaluate_polynomial(coeffs, t) for t in t_fit_full], dtype=float)

    # Nulldurchgang des Fits (erstes y<=0) + lineare Interpolation
    idx = np.where(y_fit_full <= 0.0)[0]
    if idx.size:
        k = int(idx[0])
        if k == 0:
            t_zero = t_fit_full[0]
        else:
            t1, y1 = t_fit_full[k-1], y_fit_full[k-1]
            t2, y2 = t_fit_full[k],   y_fit_full[k]
            t_zero = t1 + (0.0 - y1) * (t2 - t1) / (y2 - y1) if (y2 - y1) != 0 else t2
    else:
        t_zero = t_end  # kein Nulldurchgang im Datenbereich

    # Segment für den Fit (dünn, nur peak_time..t_zero)
    if t_zero > peak_time:
        sel = (t_fit_full >= peak_time) & (t_fit_full <= t_zero)
        t_fit_seg = t_fit_full[sel]
        y_fit_seg = y_fit_full[sel]
    else:
        t_fit_seg = np.array([peak_time, peak_time])
        y_fit_seg = np.array([y_fit_at_peak, y_fit_at_peak])

    # --- X-Achse (Initialansicht) ---
    rel_window = results.get("relevant_time_window", (rated_t, rated_t + 10.0))
    if xlim is not None:
        x_left, x_right = float(xlim[0]), float(xlim[1])
    elif view == "full":
        x_left, x_right = max(t0, rated_t), max(t_zero, rated_t + 1.0)
    else:  # "window" (Default)
        x_left, x_right = float(rel_window[0]), float(rel_window[1])

    # --- Figure/Axes ---
    fig = Figure(figsize=(9.6, 6.6), dpi=100)
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212, sharex=ax_top)

    # -------- oberer Subplot: U(t) --------
    ax_top.set_xlim((x_left, x_right))

    # y-Limits aus Daten im sichtbaren Bereich
    mask_top = (t_data >= x_left) & (t_data <= x_right)
    if np.any(mask_top):
        vmin = float(np.min(v_data[mask_top]))
        vmax = float(np.max(v_data[mask_top]))
        pad  = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
        ax_top.set_ylim(vmin - pad, vmax + pad)

    # Originalsignal
    ax_top.plot(t_data, v_data, label="Signal", linewidth=1.5)

    # Unloading-Fit (dünn)
    ax_top.plot(t_fit_seg, y_fit_seg, label="Unloading-Fit", linewidth=1.0)

    # U3: horizontale Linien
    ax_top.axhline(y_fit_at_peak, linestyle="--", linewidth=1.0, label="Fit @ Peak")
    ax_top.axhline(y_fit_plus_u3, linestyle="--", linewidth=1.0, label="Fit @ Peak + U3")

    # Vertikale Linie am Peak
    ax_top.axvline(peak_time, linestyle="--", linewidth=1.0, label="Peak-Zeit")

    ax_top.set_ylabel("U (V)")
    ax_top.grid(True, alpha=0.35)
    ax_top.legend(loc="upper right")

    # -------- unterer Subplot: dU/dt --------
    der_times = t_data
    der_raw   = np.asarray(signal.data["derivative"], dtype=float)
    ax_bot.plot(der_times, der_raw, label="dU/dt", linewidth=1.2)

    ma_label = "MA(dU/dt)"
    n = params.get("derivative_smooth_n")
    if results.get("smoothed_derivative_signal", None) is not None:
        sm = results["smoothed_derivative_signal"]
        if n is not None:
            ma_label = f"MA(dU/dt) [n={n}]"
        ax_bot.plot(sm.data["time"], sm.data["value"], label=ma_label, linewidth=0.9)

    ax_bot.set_xlabel("Zeit (s)")
    ax_bot.set_ylabel("dU/dt (V/s)")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc="upper left")

    fig.suptitle(f"{results.get('file_name', 'Signal')} – Discharge & Fit (U3)")
    fig.tight_layout()
    return fig
