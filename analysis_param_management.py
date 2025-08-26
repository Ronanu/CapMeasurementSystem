# analysis_param_management.py
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Tuple, Any, Dict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import yaml

# =========================
#   Datenmodell
# =========================

@dataclass
class AnalysisParams:
    rated_voltage: float = 3.0          # eigentlich holding voltage
    sampling_interval: float = 0.01
    window_time: float = 10.0           # Zeitfenster vor rated_time für Peak-Find
    std_factor: float = 3.0             # Threshold-Multiplikator
    derivative_smooth_n: int = 8        # Moving Average Fenster
    post_peak_cut_ratio: float = 0.4    # r > ratio * peak_value
    unload_fit_order: int = 6           # nach Peak
    rated_fit_order: int = 1            # vor Peak (linear)

    # bleibt für Kompatibilität erhalten, ist aber GUI-seitig ausgegraut
    holding_fit_order: int = 0          # 0 = Mittelwert / konstante Regression

    cutaway: float = 0.2                # für Holding-Extraction
    unloading_low_high: Tuple[float, float] = (0.6, 0.95)  # Bereich für rated_time-Bestimmung (relativ)
    min_derivative_neg: float = -0.04   # Abbruchkriterium im Rückwärts-Scan


# =========================
#   YAML I/O & Sanitizing
# =========================

DEFAULT_YAML_PATH = Path("analysis_params.yaml")

def _defaults_dict() -> Dict[str, Any]:
    return asdict(AnalysisParams())

def _params_from_dict(d: Dict[str, Any]) -> AnalysisParams:
    """Robustes Mapping: unbekannte Keys ignorieren, fehlende mit Defaults füllen."""
    defaults = _defaults_dict()
    clean: Dict[str, Any] = {}
    for f in fields(AnalysisParams):
        name = f.name
        val = d.get(name, defaults[name])

        if name == "unloading_low_high":
            try:
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    val = (float(val[0]), float(val[1]))
                else:
                    raise ValueError
            except Exception:
                val = defaults[name]

        elif f.type is int:
            try:
                val = int(val)
            except Exception:
                val = defaults[name]

        elif f.type is float:
            try:
                val = float(val)
            except Exception:
                val = defaults[name]

        clean[name] = val

    return AnalysisParams(**clean)

def _sanitize(params: AnalysisParams) -> AnalysisParams:
    """Erzwingt gültige Wertebereiche, damit die Pipeline stabil bleibt."""
    # Ganzzahlen (untere Grenzen)
    params.derivative_smooth_n = max(1, int(params.derivative_smooth_n))
    params.unload_fit_order    = max(0, int(params.unload_fit_order))
    params.rated_fit_order     = max(0, int(params.rated_fit_order))
    params.holding_fit_order   = max(0, int(params.holding_fit_order))  # bleibt 0, da GUI ausgegraut

    # Positive floats
    params.window_time       = float(params.window_time) if float(params.window_time) > 0 else 10.0
    params.sampling_interval = float(params.sampling_interval) if float(params.sampling_interval) > 0 else 0.01
    params.cutaway           = float(params.cutaway) if float(params.cutaway) > 0 else 0.2
    params.std_factor        = float(params.std_factor) if float(params.std_factor) > 0 else 3.0
    params.rated_voltage     = float(params.rated_voltage) if float(params.rated_voltage) > 0 else 3.0

    # Ratio-Begrenzung
    pr = float(params.post_peak_cut_ratio)
    params.post_peak_cut_ratio = min(1.0, max(0.0, pr))

    # Tuple-Bereich
    lo, hi = params.unloading_low_high
    try:
        lo, hi = float(lo), float(hi)
    except Exception:
        lo, hi = 0.6, 0.95
    if not (0.0 <= lo < hi <= 1.0):
        lo, hi = 0.6, 0.95
    params.unloading_low_high = (lo, hi)

    # frei (kann negativ sein)
    params.min_derivative_neg = float(params.min_derivative_neg)

    return params

def load_params_from_yaml(path: Path = DEFAULT_YAML_PATH) -> AnalysisParams:
    """Lädt YAML; bei Fehlern wird Defaults-YAML erzeugt und zurückgegeben."""
    try:
        if not path.exists():
            params = _sanitize(AnalysisParams())
            save_params_to_yaml(params, path)
            return params
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _sanitize(_params_from_dict(raw))
    except Exception:
        params = _sanitize(AnalysisParams())
        try:
            save_params_to_yaml(params, path)
        except Exception:
            pass
        return params

def save_params_to_yaml(params: AnalysisParams, path: Path = DEFAULT_YAML_PATH) -> None:
    data = asdict(_sanitize(params))
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


# =========================
#   Tkinter-Dialog
# =========================

class ParamsEditor(tk.Toplevel):
    """
    Nicht-blockierender Toplevel-Dialog.
      dlg = ParamsEditor(parent, initial_params)
      parent.wait_window(dlg)
      if dlg.result: params = dlg.result
    """
    def __init__(self, parent: tk.Misc, initial: AnalysisParams, yaml_path: Path = DEFAULT_YAML_PATH):
        super().__init__(parent)
        self.title(f"Analyse-Parameter — {yaml_path}")
        self.transient(parent)
        self.grab_set()
        self.yaml_path = yaml_path
        self.result: AnalysisParams | None = None

        self._vars: Dict[str, tk.Variable] = {}
        self._build_ui(initial)

        # Shortcuts
        self.bind("<Return>", lambda e: self._on_apply())
        self.bind("<Escape>", lambda e: self.destroy())

    # ---------- UI-Build ----------

    def _build_ui(self, initial: AnalysisParams):
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Validierungs-Callbacks
        vcmd_float = (self.register(self._validate_float), "%P")
        vcmd_int   = (self.register(self._validate_int), "%P")

        row = 0
        # Helper: Float-Entry
        def add_float(name: str, label: str, value: float, width: int = 12, hint: str | None = None):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            var = tk.StringVar(value=str(value))
            self._vars[name] = var
            e = ttk.Entry(frm, textvariable=var, width=width, validate="key", validatecommand=vcmd_float)
            e.grid(row=row, column=1, sticky="we", padx=2, columnspan=2)
            if hint:
                e.tooltip = hint  # optional placeholder for custom tooltip systems
            row += 1

        # Helper: Int-Spinbox
        def add_int(name: str, label: str, value: int, from_: int, to_: int, state: str = "normal", hint: str | None = None):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            var = tk.StringVar(value=str(int(value)))
            self._vars[name] = var
            sp = ttk.Spinbox(frm, textvariable=var, from_=from_, to=to_, increment=1,
                             width=8, validate="key", validatecommand=vcmd_int, state=state, justify="right")
            sp.grid(row=row, column=1, sticky="w", padx=2)
            if hint:
                sp.tooltip = hint
            row += 1

        # Helper: Tuple (Low/High)
        def add_tuple_low_high(name: str, label: str, value: Tuple[float, float]):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            low_var = tk.StringVar(value=str(value[0]))
            high_var = tk.StringVar(value=str(value[1]))
            self._vars[name + "_low"] = low_var
            self._vars[name + "_high"] = high_var
            e1 = ttk.Entry(frm, textvariable=low_var, width=8, validate="key", validatecommand=vcmd_float, justify="right")
            e2 = ttk.Entry(frm, textvariable=high_var, width=8, validate="key", validatecommand=vcmd_float, justify="right")
            e1.grid(row=row, column=1, sticky="w", padx=(2, 2))
            e2.grid(row=row, column=2, sticky="w", padx=(2, 2))
            row += 1

        # --- Felder anlegen ---
        p = initial

        add_float("rated_voltage",        "rated_voltage [V]",     p.rated_voltage)
        add_float("sampling_interval",    "sampling_interval [s]", p.sampling_interval)
        add_float("window_time",          "window_time [s]",       p.window_time)
        add_float("std_factor",           "std_factor [-]",        p.std_factor)
        add_int  ("derivative_smooth_n",  "derivative_smooth_n",   p.derivative_smooth_n, from_=1, to_=1000)
        add_float("post_peak_cut_ratio",  "post_peak_cut_ratio",   p.post_peak_cut_ratio)
        add_int  ("unload_fit_order",     "unload_fit_order",      p.unload_fit_order,    from_=0, to_=20)
        add_int  ("rated_fit_order",      "rated_fit_order",       p.rated_fit_order,     from_=0, to_=20)

        # ausgegraut:
        add_int  ("holding_fit_order",    "holding_fit_order (disabled)", p.holding_fit_order,
                  from_=0, to_=0, state="disabled")

        add_float("cutaway",              "cutaway [s]",           p.cutaway)
        add_tuple_low_high("unloading_low_high", "unloading_low_high [rel]", p.unloading_low_high)
        add_float("min_derivative_neg",   "min_derivative_neg [dV/ds]", p.min_derivative_neg)

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Abbrechen", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="In YAML speichern", command=self._on_save_yaml).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Übernehmen", command=self._on_apply).grid(row=0, column=2, padx=4)

    # ---------- Validierungen ----------

    @staticmethod
    def _validate_float(content: str) -> bool:
        # erlaubt: leer (wird später gefüllt), '-', '.', '-.', 'Zahl'
        if content == "" or content == "-" or content == "." or content == "-.":
            return True
        try:
            float(content)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_int(content: str) -> bool:
        if content == "" or content == "-":
            return True
        try:
            int(content)
            return True
        except ValueError:
            return False

    # ---------- Sammeln & Aktionen ----------

    def _collect(self) -> AnalysisParams | None:
        errs = []
        data: Dict[str, Any] = {}
        defaults = _defaults_dict()

        def get_float(name: str) -> float:
            s = self._vars[name].get().strip()
            if s in ("", "-", ".", "-."):
                raise ValueError
            return float(s)

        def get_int(name: str) -> int:
            s = self._vars[name].get().strip()
            if s in ("", "-"):
                raise ValueError
            return int(s)

        # Floats
        for name in ("rated_voltage", "sampling_interval", "window_time", "std_factor",
                     "post_peak_cut_ratio", "cutaway", "min_derivative_neg"):
            try:
                data[name] = get_float(name)
            except Exception:
                errs.append(f"{name}: ungültiger Wert → Default")
                data[name] = defaults[name]

        # Ints
        for name in ("derivative_smooth_n", "unload_fit_order", "rated_fit_order"):
            try:
                data[name] = get_int(name)
            except Exception:
                errs.append(f"{name}: ungültiger Wert → Default")
                data[name] = defaults[name]

        # holding_fit_order: aus YAML/Defaults übernehmen, GUI ist disabled
        data["holding_fit_order"] = defaults["holding_fit_order"]

        # Tuple
        try:
            lo = self._vars["unloading_low_high_low"].get().strip()
            hi = self._vars["unloading_low_high_high"].get().strip()
            if lo in ("", "-", ".", "-.") or hi in ("", "-", ".", "-."):
                raise ValueError
            lo_v, hi_v = float(lo), float(hi)
            data["unloading_low_high"] = (lo_v, hi_v)
        except Exception:
            errs.append("unloading_low_high: ungültige Werte → Default")
            data["unloading_low_high"] = defaults["unloading_low_high"]

        if errs:
            messagebox.showwarning("Eingaben prüfen", "\n".join(errs), parent=self)

        try:
            return _sanitize(AnalysisParams(**data))
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Parameter nicht übernehmen:\n{e}", parent=self)
            return None

    def _on_save_yaml(self):
        params = self._collect()
        if params is None:
            return
        try:
            save_params_to_yaml(params, self.yaml_path)
            messagebox.showinfo("OK", f"Gespeichert nach {self.yaml_path}", parent=self)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte YAML nicht speichern:\n{e}", parent=self)

    def _on_apply(self):
        params = self._collect()
        if params is None:
            return
        self.result = params
        self.destroy()
