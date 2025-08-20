
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use("TkAgg")

import numpy as np

from cap_calculations import AnalysisParams, cut_and_analyze_peak, recompute_from_peak
from io_ops import load_signal, ensure_save_dir, make_save_name, save_results
from log import logger


def make_figure(signal, results: Dict[str, Any]) -> Figure:
    """Erstellt eine Matplotlib-Figure aus Originalsignal + Ergebnissen (ohne neue Analyse)."""
    peak_time = results["peak_time"]
    peak_value = results["peak_value"]
    peak_mean = results["peak_mean"]
    threshold = results["threshold"]
    rated_time = results["rated_time"]
    outliers = results.get("outliers", [])
    relevant_time_window = results.get("relevant_time_window", (rated_time, rated_time + 10.0))

    if signal.data["derivative"].isnull().all():
        signal.get_derivative()

    smoothed_der = results.get("smoothed_derivative_signal", None)

    fig = Figure(figsize=(8.5, 4.8), dpi=100)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.set_xlim(relevant_time_window)

    tmin, tmax = relevant_time_window
    times = np.asarray(signal.data["time"], dtype=float)
    values = np.asarray(signal.data["value"], dtype=float)
    mask = (times >= tmin) & (times <= tmax)
    if np.any(mask):
        vmin = float(np.min(values[mask]))
        vmax = float(np.max(values[mask]))
        margin = 0.002 * (vmax - vmin if vmax > vmin else 1.0)
        ax1.set_ylim(vmin - margin, vmax + margin)

    ax1.plot(signal.data["time"], signal.data["value"], label="Originaldaten", alpha=1.0, linewidth=1.5)
    ax2.plot(signal.data["time"], signal.data["derivative"], label="Ableitung", alpha=0.3, linewidth=0.5)

    if smoothed_der is not None:
        ax2.plot(smoothed_der.data["time"], smoothed_der.data["value"], label=f"moving average", linewidth=1.2)

    if outliers:
        outlier_times, outlier_values = zip(*outliers)
        ax1.scatter(outlier_times, outlier_values, label="Ausreißer", s=10)
        ax1.scatter(peak_time, peak_value, label="Peak", s=20)

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

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    title = f"{results.get('file_name', 'Signal')} Peak-Detection"
    fig.suptitle(title)

    fig.tight_layout()
    return fig


class SingleFileApp:
    def __init__(self, root: tk.Tk, params: AnalysisParams):
        self.root = root
        self.params = params

        self.orig_signal = None
        self.results: Optional[Dict[str, Any]] = None
        self.file_path: Optional[str] = None
        self.file_name: Optional[str] = None
        self.save_dir: Optional[str] = None

        self.root.title("Cap Unloading – Single File Analyse")
        self.root.geometry("1100x720")

        self._build_widgets()
        self._layout_widgets()

    def _build_widgets(self):
        # Top bar
        self.top_frame = ttk.Frame(self.root)
        self.btn_open = ttk.Button(self.top_frame, text="Datei öffnen", command=self.on_open_file)
        self.lbl_filename = ttk.Label(self.top_frame, text="Keine Datei geladen", width=60)
        self.lbl_status = ttk.Label(self.top_frame, text="", foreground="#555")

        # Plot and side info
        self.plot_frame = ttk.Frame(self.root)
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None

        # Right-side results panel
        self.info_frame = ttk.LabelFrame(self.root, text="Ergebnisse")
        # map of result keys to pretty labels
        self._result_fields = [
            ("holding_voltage", "Holding [V]"),
            ("rated_time", "Rated time [s]"),
            ("peak_time", "Peak time [s]"),
            ("peak_value", "Peak value [V]"),
            ("peak_mean", "Mean before peak [V]"),
            ("threshold", "Threshold [V]"),
            ("U3", "U3 [V]"),
        ]
        self.result_labels: Dict[str, ttk.Label] = {}
        row = 0
        for key, label in self._result_fields:
            ttk.Label(self.info_frame, text=label + ":").grid(row=row, column=0, sticky="w", padx=(8, 6), pady=4)
            val_label = ttk.Label(self.info_frame, text="—", width=18)
            val_label.grid(row=row, column=1, sticky="e", padx=(0, 8), pady=4)
            self.result_labels[key] = val_label
            row += 1

        # Controls
        self.ctrl_frame = ttk.Frame(self.root)
        self.lbl_peak = ttk.Label(self.ctrl_frame, text="Peak-Zeit [s]:")
        self.entry_peak = ttk.Entry(self.ctrl_frame, width=12)
        self.btn_replot = ttk.Button(self.ctrl_frame, text="Neu plotten", command=self.on_replot_clicked)
        self.btn_save = ttk.Button(self.ctrl_frame, text="Speichern", command=self.on_save_clicked)
        self.btn_reset = ttk.Button(self.ctrl_frame, text="Reset Peak", command=self.on_reset_peak, state="disabled")

    def _layout_widgets(self):
        # Grid
        self.root.columnconfigure(0, weight=1)  # plot column grows
        self.root.columnconfigure(1, weight=0)  # info panel fixed
        self.root.rowconfigure(1, weight=1)     # central row grows

        # Top
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=8)
        self.top_frame.columnconfigure(2, weight=1)
        self.btn_open.grid(row=0, column=0, padx=(0, 8))
        self.lbl_filename.grid(row=0, column=1, sticky="w")
        self.lbl_status.grid(row=0, column=2, sticky="e")

        # Center: plot (left) and info (right)
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=6)
        self.info_frame.grid(row=1, column=1, sticky="ns", padx=(6, 10), pady=6)

        # Bottom controls
        self.ctrl_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=8)
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
            self.orig_signal = load_signal(path, sampling_interval=self.params.sampling_interval)
            self.file_path = path
            self.file_name = path.split("/")[-1]
            base_dir = "/".join(path.split("/")[:-1])
            self.save_dir = ensure_save_dir(base_dir)

            self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)

            # Peak entry
            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{self.results['peak_time']:.6f}")

            # Render
            self._render_current_plot()
            self._update_results_panel(self.results)

            self._set_status("Signal geladen und analysiert.")
            self._set_controls_state(enabled=True)
            self.btn_reset.configure(state="normal")

            self.lbl_filename.configure(text=self.file_name)

        except Exception as e:
            logger.warning(f"Fehler beim Öffnen: {e}")
            messagebox.showerror("Fehler", f"Datei konnte nicht geladen/analysiert werden:\n{e}")
            self._set_status("Fehler beim Laden.")
            self._set_controls_state(enabled=False)

    def on_replot_clicked(self):
        """Rechnet results mit manuell eingegebener Peak-Time neu und zeichnet dann neu."""
        if self.orig_signal is None:
            return
        peak_time = self._parse_peak_entry()
        if peak_time is None:
            return
        try:
            self.results = recompute_from_peak(self.orig_signal, peak_time, self.params, base_results=self.results)
            self._render_current_plot()
            self._update_results_panel(self.results)
            self._set_status("Darstellung aktualisiert (manuelle Peak-Zeit).")
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
            fresh_results = recompute_from_peak(self.orig_signal, peak_time, self.params, base_results=self.results)
            base_name = make_save_name(self.file_name) if self.file_name else "cut"
            path = save_results(fresh_results["after_peak_signal"], fresh_results, self.save_dir, base_name)
            self._set_status(f"Gespeichert: {path}")
            messagebox.showinfo("Gespeichert", f"Datei gespeichert:\n{path}")
        except Exception as e:
            logger.warning(f"Speicher-Fehler: {e}")
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def on_reset_peak(self):
        """Automatische Peak-Bestimmung via Analyse und komplettes Refresh der Results/Anzeige."""
        if self.orig_signal is None:
            return
        try:
            # Vollständige Analyse erneut ausführen (ermittelt Peak automatisch)
            self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)

            # Peak Entry aktualisieren
            auto_peak = float(self.results.get("peak_time", 0.0))
            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{auto_peak:.6f}")

            # Anzeige aktualisieren
            self._render_current_plot()
            self._update_results_panel(self.results)
            self._set_status("Peak zurückgesetzt (automatisch ermittelt).")

        except Exception as e:
            logger.warning(f"Reset-Fehler: {e}")
            messagebox.showerror("Fehler", f"Reset Peak fehlgeschlagen:\n{e}")

    # ---------- Helpers ----------

    def _parse_peak_entry(self) -> Optional[float]:
        val = self.entry_peak.get().strip()
        try:
            t = float(val)
        except Exception:
            self._flash_entry_error(self.entry_peak, "Ungültige Peak-Zeit")
            return None

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

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def _update_results_panel(self, results: Dict[str, Any]):
        """Zeigt die wichtigsten Kennwerte rechts neben dem Plot an."""
        def fmt(v):
            try:
                if v is None:
                    return "—"
                if isinstance(v, (list, tuple, dict)):
                    return "…"
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return str(v)

        for key, _label in self._result_fields:
            if key in results:
                self.result_labels[key].configure(text=fmt(results[key]))
            else:
                self.result_labels[key].configure(text="—")

    def _set_status(self, text: str):
        self.lbl_status.configure(text=text)


def run_app():
    params = AnalysisParams()
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = SingleFileApp(root, params)
    root.mainloop()
