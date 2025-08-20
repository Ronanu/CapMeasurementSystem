
from __future__ import annotations

# === External/project imports (as used in original file) ===
from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal
from cap_signal_processing import polynomial_fit, evaluate_polynomial
from signal_transformations import SignalDataLoader, SignalCutter, SignalDataSaver, MovingAverageFilter
from log import logger

# === Standard library / third-party ===
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from numpy import inf
from os.path import dirname, basename, join, exists
from os import makedirs
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib embedding
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ==========================
# Parameters & Data Types
# ==========================

@dataclass
class AnalysisParams:
    rated_voltage: float = 3.0
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


# ==========================
# Loading / Orchestration
# ==========================

def load_signal(file_path: str, sampling_interval: float = 0.01):
    """Lädt CSV als SignalData (Projektklasse)."""
    data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=sampling_interval)
    return data_loader.signal_data


# ==========================
# Analysis (pure, no I/O)
# ==========================

def _compute_core_before_peak(signal, params: AnalysisParams) -> Dict[str, Any]:
    """
    Berechnungen, die unabhängig von der Peak-Time sind:
    - holding_voltage
    - rated_time (aus Vor-Peak-Unloading linearer Fit)
    - peak_detection_signal, std_dev, linear approx im Fenster
    """
    # Holding
    holding_signal = get_holding_voltage_signal(signal, rated_voltage=params.rated_voltage, cutaway=params.cutaway)
    holding_voltage = polynomial_fit(holding_signal, order=params.holding_fit_order)[0]

    # Vor-Peak-Unloading zur Bestimmung rated_time
    unloading_pre = get_unloading_signal(
        signal,
        rated_voltage=params.rated_voltage,
        low_level=params.unloading_low_high[0],
        high_level=params.unloading_low_high[1]
    )
    unload_lin = polynomial_fit(unloading_pre, order=params.rated_fit_order)
    # Gerade: y = a*t + b  => t(rated) = (U_rated - b)/a
    rated_time = (params.rated_voltage - unload_lin[1]) / unload_lin[0]

    # Fenster fürs Peak-Detection
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
    Sucht den Peak rückwärts ab (wie im Original), mit Schwellwert und Ableitungsprüfung.
    Gibt (peak_time, peak_value, outliers) zurück.
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
        # Falls kein Break: Fallback auf rated_time
        peak_time = float(rated_time)
        # nehme Signalwert an rated_time:
        # (einfachste Näherung: nächster Wert)
        idx = int(np.argmin(np.abs(signal_to_cut.data["time"] - rated_time)))
        peak_value = float(signal_to_cut.data["value"][idx])

    return peak_time, peak_value, threshold, outliers


def _post_peak_calculations(signal, peak_time: float, peak_value: float, params: AnalysisParams) -> Dict[str, Any]:
    """
    Berechnungen ab Peak:
    - after_peak_signal
    - unloading_signal (cut_by_value r > ratio*peak_value)
    - polynomial fit nach Peak
    - U3 = peak_value - f_fit(peak_time)
    - peak_mean aus [peak_time-10, peak_time-1]
    """
    after_peak_signal = SignalCutter(signal).cut_time_range((peak_time, inf))
    unloading_signal = SignalCutter(after_peak_signal).cut_by_value("r>", params.post_peak_cut_ratio * peak_value)

    # Fit nach Peak
    post_fit = polynomial_fit(unloading_signal, order=params.unload_fit_order)

    # U3
    peak_eval_value = evaluate_polynomial(post_fit, peak_time)
    u3 = float(peak_value - peak_eval_value)

    # peak_mean
    mean_window = SignalCutter(signal).cut_time_range((peak_time - 10.0, peak_time - 1.0))
    peak_mean = float(np.mean(mean_window.data["value"])) if len(mean_window.data["value"]) > 0 else float('nan')

    # derivative (für Plotten nützlich)
    after_peak_signal.get_derivative()

    return {
        "after_peak_signal": after_peak_signal,
        "unloading_signal": unloading_signal,
        "post_peak_unloading_fit": post_fit,
        "U3": u3,
        "peak_mean": peak_mean,
    }


def cut_and_analyze_peak(signal, params: AnalysisParams) -> Tuple[object, Dict[str, Any]]:
    """
    Reine Analysepipeline, ohne I/O.
    Gibt (orig_signal, results_dict) zurück.
    """
    file_name = getattr(signal, "name", "Signal")
    logger.info(f"Analyse gestartet: {file_name}")

    pre = _compute_core_before_peak(signal, params)
    peak_time, peak_value, threshold, outliers = _find_peak_backward(
        signal, pre["rated_time"], pre["std_dev"], pre["peak_linear_function"], params
    )
    logger.info(f"Peak gefunden bei t={peak_time:.3f}s, Wert={peak_value:.3f}")

    post = _post_peak_calculations(signal, peak_time, peak_value, params)

    # smoothed derivative (nur für Plot)
    if hasattr(signal, "data") and "derivative" in signal.data and signal.data["derivative"].isnull().all():
        signal.get_derivative()
    smoothed_derivative_signal = MovingAverageFilter(signal.get_derivative_signal(), window_size=params.derivative_smooth_n).signal_data

    # Plot-Zeitfenster
    time_range = abs(peak_time - pre["rated_time"]) * 2.0
    relevant_time_window = (pre["rated_time"], pre["rated_time"] + time_range)

    results: Dict[str, Any] = {
        # Grundwerte
        "holding_voltage": pre["holding_voltage"],
        "rated_time": pre["rated_time"],
        "peak_time": peak_time,
        "peak_value": peak_value,
        "peak_mean": post["peak_mean"],
        "threshold": threshold,
        "U3": post["U3"],
        # Fits
        "pre_peak_unloading_fit": pre["peak_linear_function"],  # lineare Approx aus Fenster
        "post_peak_unloading_fit": post["post_peak_unloading_fit"],
        # Segmente
        "after_peak_signal": post["after_peak_signal"],
        "unloading_signal": post["unloading_signal"],
        "peak_detection_signal": pre["peak_detection_signal"],
        # Darstellungshilfen
        "outliers": outliers,
        "smoothed_derivative_signal": smoothed_derivative_signal,
        "relevant_time_window": relevant_time_window,
        # Metadaten
        "file_name": file_name,
        "params_used": asdict(params),
    }
    return signal, results


def recompute_from_peak(signal, peak_time: float, params: AnalysisParams, base_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Rechnet nur peak-abhängige Größen neu (für Replot/Save nach manueller Anpassung).
    Nutzt, wenn vorhanden, bestehende Basiswerte (rated_time, holding_voltage, etc.).
    """
    # Basisteile wiederverwenden falls vorhanden
    if base_results is None:
        # Falls nichts vorhanden ist, rechnen wir die pre-Peak-Kerne neu.
        pre = _compute_core_before_peak(signal, params)
        rated_time = pre["rated_time"]
        holding_voltage = pre["holding_voltage"]
        peak_detection_signal = pre["peak_detection_signal"]
        std_dev = pre["std_dev"]
        peak_linear_function = pre["peak_linear_function"]
    else:
        rated_time = base_results.get("rated_time")
        holding_voltage = base_results.get("holding_voltage")
        peak_detection_signal = base_results.get("peak_detection_signal")
        std_dev = float(np.std(peak_detection_signal.data["value"])) if peak_detection_signal is not None else 0.0
        peak_linear_function = base_results.get("pre_peak_unloading_fit")

    # peak_value aus Originalsignal an dieser Zeit
    # Suche nächsten Stützpunkt
    # Hinweis: wir nutzen das gesamte Signal (nicht geschnitten)
    times = np.asarray(signal.data["time"], dtype=float)
    values = np.asarray(signal.data["value"], dtype=float)
    idx = int(np.argmin(np.abs(times - peak_time)))
    peak_value = float(values[idx])

    post = _post_peak_calculations(signal, peak_time, peak_value, params)

    # smoothed derivative (nur für Plot)
    if hasattr(signal, "data") and "derivative" in signal.data and signal.data["derivative"].isnull().all():
        signal.get_derivative()
    smoothed_derivative_signal = MovingAverageFilter(signal.get_derivative_signal(), window_size=params.derivative_smooth_n).signal_data

    # Plot-Zeitfenster aktualisieren
    time_range = abs(peak_time - rated_time) * 2.0
    relevant_time_window = (rated_time, rated_time + time_range)

    results = {
        # Grundwerte
        "holding_voltage": holding_voltage,
        "rated_time": rated_time,
        "peak_time": float(peak_time),
        "peak_value": peak_value,
        "peak_mean": post["peak_mean"],
        "threshold": params.std_factor * (float(np.std(peak_detection_signal.data["value"])) if peak_detection_signal is not None else 0.0),
        "U3": post["U3"],
        # Fits
        "pre_peak_unloading_fit": peak_linear_function,
        "post_peak_unloading_fit": post["post_peak_unloading_fit"],
        # Segmente
        "after_peak_signal": post["after_peak_signal"],
        "unloading_signal": post["unloading_signal"],
        "peak_detection_signal": peak_detection_signal,
        # Darstellungshilfen
        "outliers": base_results.get("outliers") if base_results else [],
        "smoothed_derivative_signal": smoothed_derivative_signal,
        "relevant_time_window": relevant_time_window,
        # Metadaten
        "file_name": base_results.get("file_name") if base_results else getattr(signal, "name", "Signal"),
        "params_used": asdict(params),
    }
    return results


# ==========================
# Plotting (pure display)
# ==========================

def make_figure(signal, results: Dict[str, Any]) -> Figure:
    """Erstellt eine Matplotlib-Figure aus Originalsignal + Ergebnissen (ohne neue Analyse)."""
    peak_time = results["peak_time"]
    peak_value = results["peak_value"]
    peak_mean = results["peak_mean"]
    threshold = results["threshold"]
    rated_time = results["rated_time"]
    outliers = results.get("outliers", [])
    relevant_time_window = results.get("relevant_time_window", (rated_time, rated_time + 10.0))

    # Ableitung sicherstellen
    if signal.data["derivative"].isnull().all():
        signal.get_derivative()

    # smoothed derivative ggf. vorhanden
    smoothed_der = results.get("smoothed_derivative_signal", None)

    fig = Figure(figsize=(8.5, 4.8), dpi=100)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    # Achsenbereiche
    ax1.set_xlim(relevant_time_window)

    # Für y-Limits: Werte im Fenster holen
    tmin, tmax = relevant_time_window
    times = np.asarray(signal.data["time"], dtype=float)
    values = np.asarray(signal.data["value"], dtype=float)
    mask = (times >= tmin) & (times <= tmax)
    if np.any(mask):
        vmin = float(np.min(values[mask]))
        vmax = float(np.max(values[mask]))
        margin = 0.002 * (vmax - vmin if vmax > vmin else 1.0)
        ax1.set_ylim(vmin - margin, vmax + margin)

    # Plots
    ax1.plot(signal.data["time"], signal.data["value"], label="Originaldaten", alpha=1.0, linewidth=1.5)
    ax2.plot(signal.data["time"], signal.data["derivative"], label="Ableitung", alpha=0.3, linewidth=0.5)

    if smoothed_der is not None:
        ax2.plot(smoothed_der.data["time"], smoothed_der.data["value"], label=f"moving average", linewidth=1.2)

    # Ausreißer und Peak markieren
    if outliers:
        outlier_times, outlier_values = zip(*outliers)
        ax1.scatter(outlier_times, outlier_values, label="Ausreißer", s=10)
        ax1.scatter(peak_time, peak_value, label="Peak", s=20)

    # Linien und Beschriftung
    ax1.axvline(peak_time, linestyle='--', label="Peak-Zeitpunkt")
    ax1.axhline(peak_value, linestyle='--', label="Peak-Wert")
    if not np.isnan(peak_mean):
        ax1.axhline(peak_mean, linestyle='--', label="Peak-Mittelwert")
        ax1.axhline(peak_mean - threshold, linestyle='--', label="Threshold")

    ax1.set_xlabel("Zeit (s)")
    ax1.set_ylabel("Signalwert (V)")
    ax2.set_ylabel("Ableitung (V/s)")

    ax1.grid(True)
    ax2.grid(False)

    # Legenden
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    title = f"{results.get('file_name', 'Signal')} Peak-Detection"
    fig.suptitle(title)

    fig.tight_layout()
    return fig


# ==========================
# Saving (I/O only)
# ==========================

def make_save_name(file_name: str) -> str:
    name_parts = file_name.split('_')[:-1]
    name_parts = [n for n in name_parts if n != 'Testaufbau']
    name_parts.append('cut')
    return '_'.join(name_parts)


def save_results(after_peak_signal, results: Dict[str, Any], save_dir: str, base_name: str) -> str:
    if not exists(save_dir):
        makedirs(save_dir)

    save_path = join(save_dir, base_name + ".csv")

    header = {
        'holding_voltage': results.get('holding_voltage'),
        'unloading_parameter': results.get('post_peak_unloading_fit'),
        'peak_time': results.get('peak_time'),
        'peak_value': results.get('peak_value'),
        'peak_mean': results.get('peak_mean'),
        'plus_minus_toleranz': results.get('threshold'),
        'U3': results.get('U3'),
    }

    saver = SignalDataSaver(
        signal_data=after_peak_signal,
        filename=save_path,
        header_info=header
    )
    saver.save_to_csv()
    logger.info(f"Gespeichert: {save_path}")
    return save_path


# ==========================
# GUI (Single File only)
# ==========================

class SingleFileApp:
    def __init__(self, root: tk.Tk, params: AnalysisParams):
        self.root = root
        self.params = params

        # State
        self.orig_signal = None
        self.results: Optional[Dict[str, Any]] = None
        self.file_path: Optional[str] = None
        self.file_name: Optional[str] = None
        self.save_dir: Optional[str] = None

        # --- UI ---
        self.root.title("Cap Unloading – Single File Analyse")
        self.root.geometry("1000x700")

        self._build_widgets()
        self._layout_widgets()

    def _build_widgets(self):
        # Top-Leiste
        self.top_frame = ttk.Frame(self.root)
        self.btn_open = ttk.Button(self.top_frame, text="Datei öffnen", command=self.on_open_file)
        self.lbl_filename = ttk.Label(self.top_frame, text="Keine Datei geladen", width=60)
        self.lbl_status = ttk.Label(self.top_frame, text="", foreground="#555")

        # Plotbereich
        self.plot_frame = ttk.Frame(self.root)
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None

        # Steuerbereich
        self.ctrl_frame = ttk.Frame(self.root)
        self.lbl_peak = ttk.Label(self.ctrl_frame, text="Peak-Zeit [s]:")
        self.entry_peak = ttk.Entry(self.ctrl_frame, width=12)
        self.btn_replot = ttk.Button(self.ctrl_frame, text="Neu plotten", command=self.on_replot_clicked)
        self.btn_save = ttk.Button(self.ctrl_frame, text="Speichern", command=self.on_save_clicked)
        self.btn_reset = ttk.Button(self.ctrl_frame, text="Reset Peak", command=self.on_reset_peak, state="disabled")

    def _layout_widgets(self):
        # Grid-Konfiguration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # Plotreihe wächst

        # Top
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        self.top_frame.columnconfigure(2, weight=1)
        self.btn_open.grid(row=0, column=0, padx=(0, 8))
        self.lbl_filename.grid(row=0, column=1, sticky="w")
        self.lbl_status.grid(row=0, column=2, sticky="e")

        # Plot
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)

        # Controls
        self.ctrl_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
        self.ctrl_frame.columnconfigure(4, weight=1)
        self.lbl_peak.grid(row=0, column=0, sticky="w")
        self.entry_peak.grid(row=0, column=1, sticky="w", padx=(6, 18))
        self.btn_replot.grid(row=0, column=2, padx=(0, 6))
        self.btn_save.grid(row=0, column=3, padx=(0, 6))
        self.btn_reset.grid(row=0, column=4, sticky="w")

        # Shortcuts
        self.root.bind("<Return>", lambda e: self.on_replot_clicked())
        self.root.bind("<Control-s>", lambda e: self.on_save_clicked())

        self._set_controls_state(enabled=False)

    def _set_controls_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.entry_peak.configure(state=state)
        self.btn_replot.configure(state=state)
        self.btn_save.configure(state=state)
        self.btn_reset.configure(state=state)

    # ---------- Callbacks ----------

    def on_open_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        try:
            # Laden
            self.orig_signal = load_signal(path, sampling_interval=self.params.sampling_interval)
            self.file_path = path
            self.file_name = basename(path)
            self.save_dir = join(dirname(path), "cut_data")
            self.lbl_filename.configure(text=self.file_name)

            # Analyse
            self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)

            # Peak ins Entry
            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{self.results['peak_time']:.6f}")

            # Plot rendern
            self._render_current_plot()

            self._set_status("Signal geladen und analysiert.")
            self._set_controls_state(enabled=True)
            self.btn_reset.configure(state="normal")

        except Exception as e:
            logger.warning(f"Fehler beim Öffnen: {e}")
            messagebox.showerror("Fehler", f"Datei konnte nicht geladen/analysiert werden:\n{e}")
            self._set_status("Fehler beim Laden.")
            self._set_controls_state(enabled=False)

    def on_replot_clicked(self):
        if self.orig_signal is None:
            return
        peak_time = self._parse_peak_entry()
        if peak_time is None:
            return
        try:
            self.results = recompute_from_peak(self.orig_signal, peak_time, self.params, base_results=self.results)
            self._render_current_plot()
            self._set_status("Darstellung aktualisiert.")
        except Exception as e:
            logger.warning(f"Replot-Fehler: {e}")
            messagebox.showerror("Fehler", f"Neuplotten fehlgeschlagen:\n{e}")

    def on_save_clicked(self):
        if self.orig_signal is None or self.results is None:
            return
        peak_time = self._parse_peak_entry()
        if peak_time is None:
            return
        try:
            # Für das Speichern frisch mit aktueller Peak-Zeit rechnen
            fresh_results = recompute_from_peak(self.orig_signal, peak_time, self.params, base_results=self.results)
            base_name = make_save_name(self.file_name) if self.file_name else "cut"
            path = save_results(fresh_results["after_peak_signal"], fresh_results, self.save_dir, base_name)
            self._set_status(f"Gespeichert: {path}")
            messagebox.showinfo("Gespeichert", f"Datei gespeichert:\n{path}")
        except Exception as e:
            logger.warning(f"Speicher-Fehler: {e}")
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def on_reset_peak(self):
        """Setzt die Peak-Zeit im Entry auf den automatisch ermittelten Wert zurück und zeichnet neu."""
        if self.results is None:
            return
        try:
            auto_peak = float(self.results.get("peak_time", 0.0))
            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{auto_peak:.6f}")
            self.on_replot_clicked()
        except Exception as e:
            logger.warning(f"Reset-Fehler: {e}")

    # ---------- Helpers ----------

    def _parse_peak_entry(self) -> Optional[float]:
        val = self.entry_peak.get().strip()
        try:
            t = float(val)
        except Exception:
            self._flash_entry_error(self.entry_peak, "Ungültige Peak-Zeit")
            return None

        # optionale Bereichsprüfung
        if self.orig_signal is not None:
            tmin = float(np.min(self.orig_signal.data["time"]))
            tmax = float(np.max(self.orig_signal.data["time"]))
            if not (tmin <= t <= tmax):
                self._flash_entry_error(self.entry_peak, f"Peak-Zeit außerhalb [{tmin:.3f}, {tmax:.3f}]")
                return None
        return t

    def _flash_entry_error(self, entry: ttk.Entry, msg: str):
        self._set_status(msg)
        try:
            entry.configure(foreground="red")
            self.root.after(1200, lambda: entry.configure(foreground="black"))
        except Exception:
            pass

    def _render_current_plot(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar is not None:
            self.toolbar.destroy()
            self.toolbar = None

        fig = make_figure(self.orig_signal, self.results)
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Optional Toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def _set_status(self, text: str):
        self.lbl_status.configure(text=text)


def process_single_file_gui():
    params = AnalysisParams()  # Defaults wie oben
    root = tk.Tk()
    # ttk-Theme auf "clam" setzen für konsistente Darstellung
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = SingleFileApp(root, params)
    root.mainloop()


if __name__ == "__main__":
    process_single_file_gui()
