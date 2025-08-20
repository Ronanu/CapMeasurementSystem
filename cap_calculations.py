
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from numpy import inf

from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal
from cap_signal_processing import polynomial_fit, evaluate_polynomial
from signal_transformations import SignalCutter, MovingAverageFilter
from log import logger


@dataclass
class AnalysisParams:
    rated_voltage: float = 3.0   # eigentlich holding voltage
    sampling_interval: float = 0.01
    window_time: float = 10.0            # Zeitfenster vor rated_time für Peak-Find
    std_factor: float = 3.0              # Threshold-Multiplikator
    derivative_smooth_n: int = 8         # Moving Average Fenster
    post_peak_cut_ratio: float = 0.4     # r > ratio * peak_value
    unload_fit_order: int = 3            # nach Peak
    rated_fit_order: int = 1             # vor Peak (linear)
    holding_fit_order: int = 0           # 0 = Mittelwert / konstante Regression
    cutaway: float = 0.2                 # für Holding-Extraction
    unloading_low_high: Tuple[float, float] = (0.6, 0.90)  # Bereich für rated_time-Bestimmung
    min_derivative_neg: float = -0.04    # Abbruchkriterium im Rückwärts-Scan


def _compute_core_before_peak(signal, params: AnalysisParams) -> Dict[str, Any]:
    """Berechnungen, die unabhängig von der Peak-Time sind."""
    holding_signal = get_holding_voltage_signal(signal, rated_voltage=params.rated_voltage, cutaway=params.cutaway)
    holding_voltage = polynomial_fit(holding_signal, order=params.holding_fit_order)[0]

    unloading_pre = get_unloading_signal(
        signal,
        rated_voltage=params.rated_voltage,
        low_level=params.unloading_low_high[0],
        high_level=params.unloading_low_high[1]
    )
    unload_lin = polynomial_fit(unloading_pre, order=params.rated_fit_order)
    rated_time = (params.rated_voltage - unload_lin[1]) / unload_lin[0]

    peak_detection_signal = SignalCutter(signal).cut_time_range((rated_time - params.window_time, rated_time))
    std_dev = float(np.std(peak_detection_signal.data["value"]))
    peak_linear_function = polynomial_fit(peak_detection_signal, order=1)

    return {
        "holding_voltage": float(holding_voltage),
        "rated_time": float(rated_time),
        "peak_detection_signal": peak_detection_signal,
        "std_dev": std_dev,
        "peak_linear_function": peak_linear_function,
    }


def _find_peak_backward(signal, rated_time: float, std_dev: float, peak_linear_function, params: AnalysisParams):
    """
    Sucht den Peak rückwärts ab, mit Schwellwert und Ableitungsprüfung.
    Gibt (peak_time, peak_value, threshold, outliers) zurück.
    """
    signal_to_cut = SignalCutter(signal).cut_time_range((rated_time - params.window_time, inf))
    signal_to_cut.get_derivative()

    limit_reached = False
    threshold = params.std_factor * std_dev
    outliers: List[Tuple[float, float]] = []

    for t, val, dval in zip(
        reversed(signal_to_cut.data["time"]),
        reversed(signal_to_cut.data["value"]),
        reversed(signal_to_cut.data["derivative"])
    ):
        limit_value = evaluate_polynomial(peak_linear_function, t)
        if not limit_reached and val > limit_value - threshold:
            limit_reached = True
            outliers.append((t, val))
        if limit_reached:
            if dval < params.min_derivative_neg:
                outliers.append((t, val))
                continue
            else:
                peak_time = float(t)
                peak_value = float(val)
                break
    else:
        # Fallback: rated_time als Peak
        peak_time = float(rated_time)
        idx = int(np.argmin(np.abs(signal_to_cut.data["time"] - rated_time)))
        peak_value = float(signal_to_cut.data["value"][idx])

    return peak_time, peak_value, threshold, outliers


def _post_peak_calculations(signal, peak_time: float, peak_value: float, params: AnalysisParams) -> Dict[str, Any]:
    """
    Berechnungen ab Peak: Segmente, Fits, U3, peak_mean. 
    """
    after_peak_signal = SignalCutter(signal).cut_time_range((peak_time, inf))
    unloading_signal = SignalCutter(after_peak_signal).cut_by_value("r>", params.post_peak_cut_ratio * peak_value)

    post_fit = polynomial_fit(unloading_signal, order=params.unload_fit_order)

    peak_eval_value = evaluate_polynomial(post_fit, peak_time)
    u3 = float(peak_value - peak_eval_value)

    mean_window = SignalCutter(signal).cut_time_range((float(peak_time) - 10.0, float(peak_time) - 1.0))
    peak_mean = float(np.mean(mean_window.data["value"])) if len(mean_window.data["value"]) > 0 else float('nan')

    after_peak_signal.get_derivative()

    return {
        "after_peak_signal": after_peak_signal,
        "unloading_signal": unloading_signal,
        "post_peak_unloading_fit": post_fit,
        "U3": u3,
        "peak_mean": peak_mean,
    }


def cut_and_analyze_peak(signal, params: AnalysisParams):
    """Reine Analysepipeline, ohne I/O. Gibt (orig_signal, results_dict) zurück."""
    file_name = getattr(signal, "name", "Signal")
    logger.info(f"Analyse gestartet: {file_name}")

    pre = _compute_core_before_peak(signal, params)
    peak_time, peak_value, threshold, outliers = _find_peak_backward(
        signal, pre["rated_time"], pre["std_dev"], pre["peak_linear_function"], params
    )
    logger.info(f"Peak gefunden bei t={peak_time:.3f}s, Wert={peak_value:.3f}")

    post = _post_peak_calculations(signal, peak_time, peak_value, params)

    # Derivative sicherstellen und glätten für Plot
    if hasattr(signal, "data") and "derivative" in signal.data and signal.data["derivative"].isnull().all():
        signal.get_derivative()
    smoothed_derivative_signal = MovingAverageFilter(signal.get_derivative_signal(), window_size=params.derivative_smooth_n).signal_data

    time_range = abs(peak_time - pre["rated_time"]) * 2.0
    relevant_time_window = (pre["rated_time"], pre["rated_time"] + time_range)

    results: Dict[str, Any] = {
        "holding_voltage": pre["holding_voltage"],
        "rated_time": pre["rated_time"],
        "peak_time": peak_time,
        "peak_value": peak_value,
        "peak_mean": post["peak_mean"],
        "threshold": threshold,
        "U3": post["U3"],
        "pre_peak_unloading_fit": pre["peak_linear_function"],
        "post_peak_unloading_fit": post["post_peak_unloading_fit"],
        "after_peak_signal": post["after_peak_signal"],
        "unloading_signal": post["unloading_signal"],
        "peak_detection_signal": pre["peak_detection_signal"],
        "outliers": outliers,
        "smoothed_derivative_signal": smoothed_derivative_signal,
        "relevant_time_window": relevant_time_window,
        "file_name": file_name,
        "params_used": asdict(params),
    }
    return signal, results


def recompute_from_peak(signal, peak_time: float, params: AnalysisParams, base_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rechnet nur peak-abhängige Größen neu (für Replot/Save nach manueller Anpassung)."""
    if base_results is None:
        pre = _compute_core_before_peak(signal, params)
        rated_time = pre["rated_time"]
        holding_voltage = pre["holding_voltage"]
        peak_detection_signal = pre["peak_detection_signal"]
        peak_linear_function = pre["peak_linear_function"]
    else:
        rated_time = base_results.get("rated_time")
        holding_voltage = base_results.get("holding_voltage")
        peak_detection_signal = base_results.get("peak_detection_signal")
        peak_linear_function = base_results.get("pre_peak_unloading_fit")

    # Peak-Wert am nächstliegenden Stützpunkt
    times = np.asarray(signal.data["time"], dtype=float)
    values = np.asarray(signal.data["value"], dtype=float)
    idx = int(np.argmin(np.abs(times - peak_time)))
    peak_value = float(values[idx])

    post = _post_peak_calculations(signal, peak_time, peak_value, params)

    if hasattr(signal, "data") and "derivative" in signal.data and signal.data["derivative"].isnull().all():
        signal.get_derivative()
    smoothed_derivative_signal = MovingAverageFilter(signal.get_derivative_signal(), window_size=params.derivative_smooth_n).signal_data

    time_range = abs(float(peak_time) - float(rated_time)) * 2.0
    relevant_time_window = (rated_time, rated_time + time_range)

    if peak_detection_signal is not None:
        std_dev = float(np.std(peak_detection_signal.data["value"]))
    else:
        std_dev = 0.0
    threshold = params.std_factor * std_dev

    results = {
        "holding_voltage": holding_voltage,
        "rated_time": rated_time,
        "peak_time": float(peak_time),
        "peak_value": peak_value,
        "peak_mean": post["peak_mean"],
        "threshold": threshold,
        "U3": post["U3"],
        "pre_peak_unloading_fit": peak_linear_function,
        "post_peak_unloading_fit": post["post_peak_unloading_fit"],
        "after_peak_signal": post["after_peak_signal"],
        "unloading_signal": post["unloading_signal"],
        "peak_detection_signal": peak_detection_signal,
        "outliers": base_results.get("outliers") if base_results else [],
        "smoothed_derivative_signal": smoothed_derivative_signal,
        "relevant_time_window": relevant_time_window,
        "file_name": base_results.get("file_name") if base_results else getattr(signal, "name", "Signal"),
        "params_used": asdict(params),
    }
    return results
