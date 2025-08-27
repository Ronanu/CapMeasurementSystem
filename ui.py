# ui.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use("TkAgg")

import numpy as np

# ALT: from cap_calculations import AnalysisParams, cut_and_analyze_peak, recompute_from_peak
from cap_calculations import cut_and_analyze_peak, recompute_from_peak
# NEU: Param-Management zentralisiert
from analysis_param_management import (
    AnalysisParams, load_params_from_yaml, ParamsEditor, DEFAULT_YAML_PATH
)

from io_ops import load_signal, ensure_save_dir, make_save_name, save_results
from gui_fig import DischargePlot
from log import logger

# >>> Neu: Dateiname-Infos extrahieren
from getFileNameInfos import getFileNameInfos


class SingleFileApp:
    def __init__(self, root: tk.Tk, params: AnalysisParams):
        self.root = root
        self.params = params

        self.orig_signal = None
        self.results: Optional[Dict[str, Any]] = None
        self.file_path: Optional[str] = None
        self.file_name: Optional[str] = None
        self.save_dir: Optional[str] = None

        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        self.plot: Optional[DischargePlot] = None

        self.root.title("Cap Unloading – Single File Analyse")
        self.root.geometry("1280x860")

        self._build_widgets()
        self._layout_widgets()
        self._bind_shortcuts()

        self._set_controls_state(enabled=False)

    # ---------- UI Aufbau ----------

    def _build_widgets(self):
        # Top bar
        self.top_frame = ttk.Frame(self.root)
        self.btn_open = ttk.Button(self.top_frame, text="Datei öffnen", command=self.on_open_file)
        self.lbl_filename = ttk.Label(self.top_frame, text="Keine Datei geladen", width=60)
        self.lbl_status = ttk.Label(self.top_frame, text="", foreground="#555")

        # NEU: Button zum Öffnen des Param-Editors
        self.btn_params = ttk.Button(self.top_frame, text="Analyse-Parameter", command=self.on_open_params)

        # Center: plot + rechte Seitenleiste (Datei-Infos + Ergebnisse)
        self.plot_frame = ttk.Frame(self.root)

        # Seitenpanel
        self.side_panel = ttk.Frame(self.root)

        # ---- Abschnitt 1: Dateiname-Infos
        self.fileinfo_frame = ttk.LabelFrame(self.side_panel, text="Dateiname-Infos")
        self._fileinfo_fields = [
            ("manufacturer", "Hersteller"),
            ("capacitance", "Kapazität [F]"),
            ("typ", "Typ"),
            ("methode", "Methode"),
            ("klass", "Klasse"),
            ("dut", "DUT"),
            ("version", "Version"),
        ]
        self.fileinfo_labels: Dict[str, ttk.Label] = {}
        for r, (key, label) in enumerate(self._fileinfo_fields):
            ttk.Label(self.fileinfo_frame, text=label + ":").grid(row=r, column=0, sticky="w", padx=(8, 6), pady=4)
            val_label = ttk.Label(self.fileinfo_frame, text="—", width=18)
            val_label.grid(row=r, column=1, sticky="e", padx=(0, 8), pady=4)
            self.fileinfo_labels[key] = val_label

        # ---- Abschnitt 2: Ergebnisse
        self.info_frame = ttk.LabelFrame(self.side_panel, text="Ergebnisse")
        self._result_fields = [
            ("holding_voltage", "Holding [V]"),
            ("approx_peak_time", "Rated time [s]"),
            ("peak_time", "Peak time [s]"),
            ("peak_value", "Peak value [V]"),
            ("peak_mean", "Mean before peak [V]"),
            ("threshold", "Threshold [V]"),
            ("U3", "U3 [V]"),
            ("U3_mean", "U3 Mean [V]"),
        ]
        self.result_labels: Dict[str, ttk.Label] = {}
        for r, (key, label) in enumerate(self._result_fields):
            ttk.Label(self.info_frame, text=label + ":").grid(row=r, column=0, sticky="w", padx=(8, 6), pady=4)
            val_label = ttk.Label(self.info_frame, text="—", width=18)
            val_label.grid(row=r, column=1, sticky="e", padx=(0, 8), pady=4)
            self.result_labels[key] = val_label

        # Matplotlib fig elements
        self.figure = None
        self.canvas = None
        self.toolbar = None

        # --- Controls unter den Plots ---
        self.ctrl_frame = ttk.Frame(self.root)

        # Row 0: Ansichten
        self.lbl_views = ttk.Label(self.ctrl_frame, text="Ansichten:")
        self.btn_view_all = ttk.Button(self.ctrl_frame, text="View All", command=self.on_view_all)
        self.btn_view_full = ttk.Button(self.ctrl_frame, text="Full Discharge", command=self.on_view_full)
        self.btn_view_window = ttk.Button(self.ctrl_frame, text="Peak Window", command=self.on_view_window)

        # Row 1: Zoom/Pan (Zoom − vor +), daneben Peak-Operationen
        self.btn_zoom_out = ttk.Button(self.ctrl_frame, text="Zoom −", command=lambda: self._zoom(1.25))
        self.btn_zoom_in = ttk.Button(self.ctrl_frame, text="Zoom +", command=lambda: self._zoom(0.8))
        self.btn_pan_left = ttk.Button(self.ctrl_frame, text="← Pan", command=lambda: self._pan_relative(-0.2))
        self.btn_pan_right = ttk.Button(self.ctrl_frame, text="Pan →", command=lambda: self._pan_relative(+0.2))

        self.lbl_peak = ttk.Label(self.ctrl_frame, text="Peak-Zeit [s]:")
        self.entry_peak = ttk.Entry(self.ctrl_frame, width=14)
        self.btn_replot = ttk.Button(self.ctrl_frame, text="Neu plotten", command=self.on_replot_clicked)
        self.btn_reset = ttk.Button(self.ctrl_frame, text="Reset Peak", command=self.on_reset_peak)

        # Row 2: Feinjustage Peak + Speichern ganz rechts
        self.btn_peak_minus = ttk.Button(self.ctrl_frame, text="−0.01 s", command=lambda: self._nudge_peak(-0.01))
        self.btn_peak_plus = ttk.Button(self.ctrl_frame, text="+0.01 s", command=lambda: self._nudge_peak(+0.01))
        self.btn_save = ttk.Button(self.ctrl_frame, text="Speichern", command=self.on_save_clicked)

    def _layout_widgets(self):
        # Grid weights
        self.root.columnconfigure(0, weight=1)  # plot column
        self.root.columnconfigure(1, weight=0)  # side column
        self.root.rowconfigure(1, weight=1)

        # Top
        self.top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        self.top_frame.columnconfigure(3, weight=1)
        self.btn_open.grid(row=0, column=0, padx=(0, 8))
        self.lbl_filename.grid(row=0, column=1, sticky="w")
        self.btn_params.grid(row=0, column=2, padx=8)  # NEU
        self.lbl_status.grid(row=0, column=3, sticky="e")

        # Center
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=6)

        # Seitenpanel (rechts)
        self.side_panel.grid(row=1, column=1, sticky="ns", padx=(6, 10), pady=6)
        self.side_panel.columnconfigure(0, weight=1)

        self.fileinfo_frame.grid(row=0, column=0, sticky="new", pady=(0, 8))
        self.info_frame.grid(row=1, column=0, sticky="ns")

        # Controls unten
        self.ctrl_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(6, 10))
        for c in range(12):
            self.ctrl_frame.columnconfigure(c, weight=0)
        self.ctrl_frame.columnconfigure(11, weight=1)

        # --- Row 0: Ansichten ---
        r = 0
        c = 0
        self.lbl_views.grid(row=r, column=c, sticky="w", padx=(0, 6), pady=4); c += 1
        self.btn_view_all.grid(row=r, column=c, padx=4, pady=4); c += 1
        self.btn_view_full.grid(row=r, column=c, padx=4, pady=4); c += 1
        self.btn_view_window.grid(row=r, column=c, padx=4, pady=4); c += 1

        ttk.Label(self.ctrl_frame, text="\t").grid(row=r, column=c, padx=10); c += 1

        self.lbl_peak.grid(row=r, column=c, sticky="e", padx=(4, 2)); c += 1
        self.entry_peak.grid(row=r, column=c, sticky="w", padx=(6, 12)); c += 1
        self.btn_replot.grid(row=r, column=c, padx=4); c += 1
        self.btn_reset.grid(row=r, column=c, padx=4); c += 1

        # --- Row 1: Zoom/Pan + Peak-Ops ---
        r = 1
        c = 0
        self.btn_zoom_out.grid(row=r, column=c, padx=4, pady=6); c += 1
        self.btn_zoom_in.grid(row=r, column=c, padx=4, pady=6); c += 1
        self.btn_pan_left.grid(row=r, column=c, padx=4, pady=6); c += 1
        self.btn_pan_right.grid(row=r, column=c, padx=4, pady=6); c += 1

        c = c + 3
        self.btn_peak_minus.grid(row=r, column=c, padx=4, pady=(4, 6)); c += 1
        self.btn_peak_plus.grid(row=r, column=c, padx=4, pady=(4, 6)); c += 1

        ttk.Label(self.ctrl_frame, text="").grid(row=r, column=10, sticky="ew")
        self.btn_save.grid(row=r, column=11, sticky="e", padx=(4, 0), pady=(4, 6))

    def _bind_shortcuts(self):
        self.root.bind("<Return>", lambda e: self.on_replot_clicked())
        self.root.bind("<Control-s>", lambda e: self.on_save_clicked())
        self.root.bind("<Control-equal>", lambda e: self._zoom(0.8))   # Ctrl+= → zoom in
        self.root.bind("<Control-minus>", lambda e: self._zoom(1.25))  # Ctrl+- → zoom out
        self.root.bind("<Left>", lambda e: self._pan_relative(-0.2))
        self.root.bind("<Right>", lambda e: self._pan_relative(+0.2))

    def _set_controls_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (
            self.entry_peak, self.btn_replot, self.btn_reset, self.btn_save,
            self.btn_view_all, self.btn_view_full, self.btn_view_window,
            self.btn_zoom_in, self.btn_zoom_out, self.btn_pan_left, self.btn_pan_right,
            self.btn_peak_minus, self.btn_peak_plus
        ):
            w.configure(state=state)

    # ---------- Callbacks ----------

    def on_open_params(self):
        """Dialog öffnen; bei Übernahme optional aktuelle Analyse neu rechnen."""
        dlg = ParamsEditor(self.root, self.params, yaml_path=DEFAULT_YAML_PATH)
        self.root.wait_window(dlg)
        if dlg.result is None:
            return
        self.params = dlg.result
        self._set_status("Parameter übernommen.")
        # falls ein Signal geladen ist → Analyse mit neuen Parametern auffrischen
        if self.orig_signal is not None:
            try:
                self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)
                auto_peak = float(self.results.get("peak_time", 0.0))
                self.entry_peak.configure(state="normal")
                self.entry_peak.delete(0, tk.END)
                self.entry_peak.insert(0, f"{auto_peak:.2f}")
                self._render_current_plot(initial_view="window")
                self._update_results_panel(self.results)
                self._set_status("Analyse mit neuen Parametern aktualisiert.")
            except Exception as e:
                logger.warning(f"Neu-Analyse nach Param-Update fehlgeschlagen: {e}")
                messagebox.showerror("Fehler", f"Neu-Analyse fehlgeschlagen:\n{e}")

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

            # Analyse (liefert signal & results)
            self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)
            logger.debug(f"Analyseergebnisse: {self.results}")

            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{self.results['peak_time']:.2f}")

            self._update_fileinfo_panel(self.file_name)
            self._render_current_plot(initial_view="window")
            self._update_results_panel(self.results)

            self._set_status("Signal geladen und analysiert.")
            self._set_controls_state(enabled=True)
            self.lbl_filename.configure(text=self.file_name)

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
            self._render_current_plot(keep_view=True)
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
        if self.orig_signal is None:
            return
        try:
            self.orig_signal, self.results = cut_and_analyze_peak(self.orig_signal, self.params)
            auto_peak = float(self.results.get("peak_time", 0.0))
            self.entry_peak.configure(state="normal")
            self.entry_peak.delete(0, tk.END)
            self.entry_peak.insert(0, f"{auto_peak:.2f}")
            self._render_current_plot(initial_view="window")
            self._update_results_panel(self.results)
            self._set_status("Peak zurückgesetzt (automatisch ermittelt).")
        except Exception as e:
            logger.warning(f"Reset-Fehler: {e}")
            messagebox.showerror("Fehler", f"Reset Peak fehlgeschlagen:\n{e}")

    # --- View controls ---

    def on_view_window(self):
        if self.plot is None:
            return
        self.plot.zoom_window()
        self._redraw_canvas()

    def on_view_full(self):
        if self.plot is None:
            return
        self.plot.zoom_full()
        self._redraw_canvas()

    def on_view_all(self):
        if self.plot is None:
            return
        try:
            t0, t1 = self.plot._data_time_bounds()
            t_zero = self.plot._compute_t_zero()
            self.plot.set_xlim(t0, t_zero)
            self._redraw_canvas()
        except Exception as e:
            logger.warning(f"View All Fehler: {e}")

    def _zoom(self, factor: float):
        if self.plot is None:
            return
        self.plot.zoom(factor)
        self._redraw_canvas()

    def _pan_relative(self, frac: float):
        if self.plot is None:
            return
        xlim = self.plot.get_xlim()
        if not xlim:
            return
        a, b = xlim
        width = max(1e-9, b - a)
        delta = float(frac) * width
        self.plot.pan(delta)
        self._redraw_canvas()

    # --- Peak Feinjustage ---

    def _nudge_peak(self, delta: float):
        if self.orig_signal is None:
            return
        val = self.entry_peak.get().strip()
        try:
            t = float(val)
        except Exception:
            self._flash_entry_error(self.entry_peak, "Ungültige Peak-Zeit")
            return
        t_new = t + float(delta)
        tmin = float(np.min(self.orig_signal.data["time"]))
        tmax = float(np.max(self.orig_signal.data["time"]))
        t_new = max(tmin, min(tmax, t_new))
        self.entry_peak.delete(0, tk.END)
        self.entry_peak.insert(0, f"{t_new:.2f}")
        self.on_replot_clicked()

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

    def _render_current_plot(self, initial_view: str | None = None, keep_view: bool = False):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar is not None:
            self.toolbar.destroy()
            self.toolbar = None

        if self.plot is None:
            self.plot = DischargePlot(self.orig_signal, self.results)
            fig = self.plot.draw(initial_view=initial_view or "window")
        else:
            if keep_view:
                try:
                    cur = self.plot.get_view()
                except Exception:
                    cur = None
                self.plot.update_results(self.results, keep_view=True)
                if cur is not None:
                    self.plot.set_xlim(*cur["xlim"])
                fig = self.plot.fig
            else:
                self.plot = DischargePlot(self.orig_signal, self.results)
                fig = self.plot.draw(initial_view=initial_view or "window")

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def _redraw_canvas(self):
        if self.canvas is not None and self.plot is not None:
            self.canvas.draw()

    def _update_results_panel(self, results: Dict[str, Any]):
        def fmt(v):
            try:
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.6g}"
                if isinstance(v, (list, tuple, dict)):
                    return "…"
                return str(v)
            except Exception:
                return str(v)

        for key, _label in self._result_fields:
            self.result_labels.get(key, ttk.Label()).configure(text=fmt(results.get(key)))

    def _update_fileinfo_panel(self, filename: str):
        def fmt(v):
            if v is None or v == "":
                return "—"
            return str(v)
        info = {
            "manufacturer": None,
            "capacitance": None,
            "typ": None,
            "methode": None,
            "klass": None,
            "dut": None,
            "version": None,
        }
        try:
            manufacturer, capacitance, typ, methode, klass, dut, version = getFileNameInfos(filename)
            info.update({
                "manufacturer": manufacturer,
                "capacitance": capacitance,
                "typ": typ,
                "methode": methode,
                "klass": klass,
                "dut": dut,
                "version": version,
            })
        except Exception as e:
            logger.warning(f"Konnte Dateiname-Infos nicht extrahieren: {e}")

        for key, _label in self._fileinfo_fields:
            self.fileinfo_labels.get(key, ttk.Label()).configure(text=fmt(info.get(key)))

    def _set_status(self, text: str):
        self.lbl_status.configure(text=text)


def run_app():
    params = load_params_from_yaml(DEFAULT_YAML_PATH)
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = SingleFileApp(root, params)
    root.mainloop()
