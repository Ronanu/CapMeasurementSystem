# Updated make_figure per your requests:
# - Two subplots: top = voltage & fit; bottom = derivatives
# - Unloading fit drawn only from peak_time to when fit reaches 0 V (thin line)
# - Vertical line at peak_time
# - U3 shown with two horizontal lines (fit@peak and fit@peak + U3)
# - Derivative subplot: raw derivative (slightly thicker) and moving average (thin)
# - Moving average parameters rendered as text in the second subplot

def make_figure(signal, results: dict):
    """
    Returns a Matplotlib Figure with two subplots:
      (1) Top: voltage vs time with unloading fit (thin) from peak_time to fit==0,
               vertical peak line, and two horizontals for U3 visualization.
      (2) Bottom: derivative (raw) and moving-average (thin), with MA parameters annotated.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from cap_signal_processing import evaluate_polynomial

    peak_time = float(results["peak_time"])
    U3 = float(results["U3"])
    rated_time = float(results["rated_time"])
    coeffs = results["post_peak_unloading_fit"]
    relevant_time_window = results.get("relevant_time_window", (rated_time, rated_time + 10.0))

    # Ensure derivative exists
    if signal.data["derivative"].isnull().all():
        signal.get_derivative()

    smoothed_der = results.get("smoothed_derivative_signal", None)

    # Fit level at peak and U3 top line
    y_fit_at_peak = float(evaluate_polynomial(coeffs, peak_time))
    y_fit_plus_u3 = y_fit_at_peak + U3

    # Time window
    tmin, tmax = map(float, relevant_time_window)
    times = np.asarray(signal.data["time"], dtype=float)
    values = np.asarray(signal.data["value"], dtype=float)

    # Determine where the fit reaches 0 V (from peak_time forward)
    # Fallback: if never crosses, we limit to tmax
    t_grid = np.linspace(max(peak_time, tmin), max(tmax, peak_time), 1000)
    y_grid = np.array([evaluate_polynomial(coeffs, t) for t in t_grid], dtype=float)
    idx_cross = np.where(y_grid <= 0.0)[0]
    if idx_cross.size > 0:
        i = idx_cross[0]
        if i == 0:
            t_zero = t_grid[0]
        else:
            # linear interpolation for a better crossing estimate
            t1, y1 = t_grid[i-1], y_grid[i-1]
            t2, y2 = t_grid[i], y_grid[i]
            if (y2 - y1) != 0:
                t_zero = t1 + (0.0 - y1) * (t2 - t1) / (y2 - y1)
            else:
                t_zero = t2
    else:
        t_zero = max(t_grid)  # no crossing in window

    # Build figure with two rows, shared x
    fig = Figure(figsize=(9.5, 6.4), dpi=100)
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212, sharex=ax_top)

    # -------- Top subplot: Voltage --------
    ax_top.set_xlim((tmin, tmax))

    # y-limits from data within window
    mask = (times >= tmin) & (times <= tmax)
    if np.any(mask):
        vmin = float(np.min(values[mask]))
        vmax = float(np.max(values[mask]))
        margin = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
        ax_top.set_ylim(vmin - margin, vmax + margin)

    # Original signal
    ax_top.plot(times, values, label="Signal", linewidth=1.5)

    # Unloading fit (thin), only from peak_time to t_zero
    t_fit_seg = np.linspace(peak_time, t_zero, 300)
    if t_fit_seg.size >= 2:
        y_fit_seg = [evaluate_polynomial(coeffs, t) for t in t_fit_seg]
        ax_top.plot(t_fit_seg, y_fit_seg, label="Unloading-Fit", linewidth=1.0)

    # U3 horizontal lines
    ax_top.axhline(y_fit_at_peak, linestyle="--", linewidth=1.0, label="Fit @ Peak")
    ax_top.axhline(y_fit_plus_u3, linestyle="--", linewidth=1.0, label="Fit @ Peak + U3")

    # Vertical line at peak time
    ax_top.axvline(peak_time, linestyle="--", linewidth=1.0, label="Peak-Zeit")

    # Labels/legend
    ax_top.set_ylabel("U (V)")
    ax_top.grid(True, alpha=0.35)
    ax_top.legend(loc="upper right")

    # -------- Bottom subplot: Derivative --------
    der_times = np.asarray(signal.data["time"], dtype=float)
    der_values = np.asarray(signal.data["derivative"], dtype=float)
    ax_bot.plot(der_times, der_values, label="dU/dt (raw)", linewidth=1.2)

    if smoothed_der is not None:
        ax_bot.plot(smoothed_der.data["time"], smoothed_der.data["value"], label="MA(dU/dt)", linewidth=0.9)

    # Annotate moving average parameters
    params_used = results.get("params_used", {})
    ma_n = params_used.get("derivative_smooth_n", None)
    if ma_n is not None:
        ax_bot.text(
            0.99, 0.95, f"Moving Average: n={ma_n}",
            ha="right", va="top", transform=ax_bot.transAxes
        )

    # Axes labels and grid
    ax_bot.set_xlabel("Zeit (s)")
    ax_bot.set_ylabel("dU/dt (V/s)")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc="upper right")

    # Title
    fig.suptitle(f"{results.get('file_name', 'Signal')} – Discharge & Fit (U3)")

    fig.tight_layout()
    return fig

# Print function body for copy/paste
import inspect
print(inspect.getsource(make_figure))
