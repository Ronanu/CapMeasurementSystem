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
    # --- Grundlegend ---
    holding_voltage: float = 3.0            # Haltespannung (früher rated_voltage)
    sampling_interval: float = 0.01         # Zeitabstand zwischen Messpunkten [s]

    # --- Peak-Erkennung ---
    peak_search_window_s: float = 10.0      # Suchfenster vor rated_time [s]
    std_factor: float = 3.0                 # Threshold-Multiplikator (σ-Faktor)
    derivative_smooth_n: int = 8            # Fensterlänge gleitender Mittelwert für Ableitung
    min_derivative_neg: float = -0.04       # Abbruchkriterium: wie negativ darf dV/dt noch sein

    # --- Fits ---
    # rated_fit_order wird in der GUI deaktiviert & intern auf 1 erzwungen (Entlade‑Bezugslinie vor dem Peak)
    rated_fit_order: int = 1                # Polynomgrad vor dem Peak (rated_time via Schnitt mit Haltespannung)
    unload_fit_order: int = 6               # Polynomgrad nach dem Peak (Entladeast)
    # holding_fit_order bleibt 0 (Mittelwert) – GUI deaktiviert; behalten für Kompatibilität der Oberfläche
    holding_fit_order: int = 0

    # --- Bereiche / Zuschnitt ---
    unloading_low_high: Tuple[float, float] = (0.6, 0.95)  # relativer Bereich r/peak (low..high) für Nach‑Peak‑Fit
    peak_mean_window_s: Tuple[float, float] = (5.1, 0.1)   # (max_vor_peak, min_vor_peak) in Sekunden, beide positiv

    # --- Preprocessing ---
    cutaway: float = 0.2                   # Anfangssekunden ignorieren (Holding-Extraktion)

# =========================
#   YAML I/O & Sanitizing
# =========================

DEFAULT_YAML_PATH = Path("analysis_params.yaml")

def _defaults_dict() -> Dict[str, Any]:
    return asdict(AnalysisParams())

def _params_from_dict(d: Dict[str, Any]) -> AnalysisParams:
    """
    Striktes Mapping auf aktuelle Felder (keine Alt‑Keys mehr zulässig).
    Unbekannte Keys werden ignoriert; fehlende werden mit Defaults belegt.
    """
    defaults = _defaults_dict()
    clean: Dict[str, Any] = {}

    # Nur bekannte Felder lesen, nichts mappen (kein 'rated_voltage' etc.).
    for f in fields(AnalysisParams):
        name = f.name
        val = d.get(name, defaults[name])

        # Tupel-Felder validieren
        if name in ("unloading_low_high", "peak_mean_window_s"):
            try:
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    val = (float(val[0]), float(val[1]))
                else:
                    raise ValueError
            except Exception:
                val = defaults[name]

        # Int-Felder
        elif f.type is int:
            try:
                val = int(val)
            except Exception:
                val = defaults[name]

        # Float-Felder
        elif f.type is float:
            try:
                val = float(val)
            except Exception:
                val = defaults[name]

        clean[name] = val

    return AnalysisParams(**clean)

def _sanitize(params: AnalysisParams) -> AnalysisParams:
    """Zwingt gültige Wertebereiche; rated_fit_order wird fixiert."""

    # Ganzzahlen & Fixierungen
    params.derivative_smooth_n = max(1, int(params.derivative_smooth_n))
    params.unload_fit_order    = max(0, int(params.unload_fit_order))
    params.rated_fit_order     = 1  # erzwingen (GUI disabled)
    params.holding_fit_order   = 0  # beibehalten

    # Positive Floats
    params.peak_search_window_s = float(params.peak_search_window_s) if float(params.peak_search_window_s) > 0 else 10.0
    params.sampling_interval    = float(params.sampling_interval) if float(params.sampling_interval) > 0 else 0.01
    params.cutaway              = float(params.cutaway) if float(params.cutaway) > 0 else 0.2
    params.std_factor           = float(params.std_factor) if float(params.std_factor) > 0 else 3.0
    params.holding_voltage      = float(params.holding_voltage) if float(params.holding_voltage) > 0 else 3.0

    # Bereich: unloading_low_high als 0..1 mit low < high
    low, high = params.unloading_low_high
    try:
        low = float(low)
        high = float(high)
    except Exception:
        low, high = 0.6, 0.95
    low = min(max(0.0, low), 1.0)
    high = min(max(0.0, high), 1.0)
    if not (low < high):
        low, high = 0.6, 0.95
    params.unloading_low_high = (low, high)

    # Bereich: peak_mean_window_s (beide >0, max_before > min_before)
    wmax, wmin = params.peak_mean_window_s
    try:
        wmax = float(wmax)
        wmin = float(wmin)
    except Exception:
        wmax, wmin = 5.1, 0.1
    if wmax <= 0 or wmin <= 0 or (wmax <= wmin):
        wmax, wmin = 5.1, 0.1
    params.peak_mean_window_s = (wmax, wmin)

    # min_derivative_neg darf 0 oder negativ sein (streng negativ sinnvoll)
    try:
        params.min_derivative_neg = float(params.min_derivative_neg)
    except Exception:
        params.min_derivative_neg = -0.04

    return params

def load_params_from_yaml(path: Path = DEFAULT_YAML_PATH) -> AnalysisParams:
    """
    Lädt Parameter, und falls die YAML noch nicht existiert,
    wird sie automatisch mit DEFAULT-Werten erstellt.
    """
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return _sanitize(_params_from_dict(data))
        except Exception:
            # Fallback auf Defaults (und Datei neu schreiben)
            params = _sanitize(AnalysisParams())
            try:
                save_params_to_yaml(params, path)
            except Exception:
                pass
            return params
    else:
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
#   Tkinter‑Dialog
# =========================

class ParamsEditor(tk.Toplevel):
    def __init__(self, master: tk.Tk, params: AnalysisParams, yaml_path: Path = DEFAULT_YAML_PATH):
        super().__init__(master)
        self.title("Analyse‑Parameter")
        self.resizable(False, False)
        self.result: AnalysisParams | None = None
        self.yaml_path = yaml_path

        self._vars: Dict[str, tk.StringVar] = {}
        self._hint_widgets: Dict[str, tk.Widget] = {}
        self._hint_visible: Dict[str, bool] = {}
        self._build_ui(params)
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        self.transient(master)
        self.wait_visibility()
        try:
            self.focus()
        except Exception:
            pass

    # ---------- UI Aufbau ----------
    def _build_ui(self, initial: AnalysisParams):
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        frm.columnconfigure(1, weight=1)  # Eingabespalte dehnbar

        # Validierer
        vcmd_float = (self.register(self._validate_float), "%P")
        vcmd_int   = (self.register(self._validate_int), "%P")

        row = 0

        def _add_info_button(name: str, hint: str | None, row_index: int):
            """Erzeugt einen kleinen Info‑Button rechts, der den unterliegenden Hint‑Text ein/ausklappt."""
            if not hint:
                return
            btn = ttk.Button(frm, text="ℹ", width=2,
                             command=lambda n=name: self._toggle_hint(n),
                             takefocus=False)
            btn.grid(row=row_index, column=3, sticky="e", padx=(6,0))
            # vorbereiteter Hint‑Label (unter dem Feld), initial versteckt
            hint_lbl = ttk.Label(frm, text=hint, foreground="#444", wraplength=520, justify="left")
            # wir platzieren ihn auf row_index+1, über die volle Breite
            hint_lbl.grid(row=row_index+1, column=0, columnspan=4, sticky="w", pady=(0,6))
            hint_lbl.grid_remove()
            self._hint_widgets[name] = hint_lbl
            self._hint_visible[name] = False

        def add_float(name: str, label: str, value: float, width: int = 12, hint: str | None = None, state: str = "normal"):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0,10), pady=4)
            var = tk.StringVar(value=str(value))
            self._vars[name] = var
            e = ttk.Entry(frm, textvariable=var, width=width, validate="key", validatecommand=vcmd_float, state=state, justify="right")
            e.grid(row=row, column=1, sticky="we", padx=2, columnspan=2)
            _add_info_button(name, hint, row)
            row += 2  # +1 für Feldzeile, +1 für (ggf. sichtbare) Hint-Zeile

        def add_int(name: str, label: str, value: int, from_: int, to_: int, hint: str | None = None, state: str = "normal"):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0,10), pady=4)
            var = tk.StringVar(value=str(int(value)))
            self._vars[name] = var
            sp = ttk.Spinbox(frm, textvariable=var, from_=from_, to_=to_, increment=1, width=8,
                             validate="key", validatecommand=vcmd_int, state=state, justify="right")
            sp.grid(row=row, column=1, sticky="w", padx=2)
            _add_info_button(name, hint, row)
            row += 2

        def add_tuple_low_high(name: str, label: str, value: Tuple[float,float], hint: str | None = None):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=(0,10), pady=4)
            vlow = tk.StringVar(value=str(value[0]))
            vhigh = tk.StringVar(value=str(value[1]))
            self._vars[name+"_low"] = vlow
            self._vars[name+"_high"] = vhigh
            e1 = ttk.Entry(frm, textvariable=vlow, width=6, validate="key", validatecommand=vcmd_float, justify="right")
            e2 = ttk.Entry(frm, textvariable=vhigh, width=6, validate="key", validatecommand=vcmd_float, justify="right")
            e1.grid(row=row, column=1, sticky="w", padx=(0,2))
            e2.grid(row=row, column=2, sticky="w", padx=(2,0))
            _add_info_button(name, hint, row)
            row += 2

        # ---- Felder mit Erklärtexten ----

        p = initial

        add_float(
            "holding_voltage", "holding_voltage [V]", p.holding_voltage,
            hint="Haltespannung: Referenz für Holding‑Mittel und zur Bestimmung der rated time (Schnitt der Entlade‑Bezugslinie mit dieser Spannung)."
        )
        add_float(
            "sampling_interval", "sampling_interval [s]", p.sampling_interval,
            hint="Zeitabstand zwischen Messpunkten. Muss der Aufzeichnung entsprechen; beeinflusst Ableitungen und Zeiten."
        )
        add_float(
            "peak_search_window_s", "peak_search_window_s [s]", p.peak_search_window_s,
            hint="Suchfenster VOR der rated time. In diesem Bereich wird die Entlade‑Bezugslinie geschätzt und der Rückwärts‑Scan gestartet."
        )
        add_float(
            "std_factor", "std_factor [-]", p.std_factor,
            hint="Schwellwert = std_factor × σ des Signals im Suchfenster. Höher = konservativer, weniger falsche Peaks."
        )
        add_int(
            "derivative_smooth_n", "derivative_smooth_n [n]", p.derivative_smooth_n, from_=1, to_=999,
            hint="Fensterlänge für gleitenden Mittelwert der ABLEITUNG. Größer = glatter, aber träger."
        )
        add_float(
            "min_derivative_neg", "min_derivative_neg [dV/ds]", p.min_derivative_neg,
            hint="Ausstiegsschwelle im Rückwärts‑Scan: Punkte mit zu starker negativer Steigung werden verworfen; erst bei flacherer Steigung wird der Peak fixiert."
        )

        # rated_fit_order – disabled, intern fix = 1
        add_int(
            "rated_fit_order", "rated_fit_order (fix=1)", p.rated_fit_order, from_=1, to_=1, state="disabled",
            hint="Polynomgrad der Entlade‑Bezugslinie VOR dem Peak. rated time = Schnitt von m·t+b mit der Haltespannung."
        )

        add_int(
            "unload_fit_order", "unload_fit_order", p.unload_fit_order, from_=0, to_=12,
            hint="Polynomgrad NACH dem Peak (Entladeast). Höher = flexibler, Risiko Überanpassung."
        )

        add_tuple_low_high(
            "unloading_low_high", "unloading_low_high [rel]", p.unloading_low_high,
            hint="Relativer Spannungsbereich des ENTLADEVORGANGS für Nach‑Peak‑Fit: low ≤ r/peak ≤ high. Wird beidseitig geschnitten."
        )

        add_float(
            "cutaway", "cutaway [s]", p.cutaway,
            hint="Anfangsdauer, die beim Holding ignoriert wird (Schalt‑/Einschwingartefakte entfernen)."
        )

        add_tuple_low_high(
            "peak_mean_window_s", "peak_mean_window_s [s,s]", p.peak_mean_window_s,
            hint="Fenster für Peak‑Mittelwert vor dem Peak, als positive Offsets: (max_vor_peak, min_vor_peak). Beispiel: (5.1, 0.1) → [−5.1 s … −0.1 s]."
        )

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=4, sticky="e", pady=(10,0))
        ttk.Button(btns, text="Speichern", command=self._on_save).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Übernehmen", command=self._on_apply).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Abbrechen", command=self.destroy).grid(row=0, column=2, padx=4)

    # ---------- Hint-Logik ----------

    def _toggle_hint(self, name: str):
        w = self._hint_widgets.get(name)
        if not w:
            return
        visible = self._hint_visible.get(name, False)
        if visible:
            w.grid_remove()
            self._hint_visible[name] = False
        else:
            w.grid()
            self._hint_visible[name] = True

    # ---------- Validierung ----------

    @staticmethod
    def _validate_float(content: str) -> bool:
        if content in ("", "-", ".", "-."):
            return True
        try:
            float(content)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_int(content: str) -> bool:
        if content in ("", "-"):
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
            return int(float(s))

        def get_pair(name: str) -> Tuple[float, float]:
            s1 = self._vars[name+"_low"].get().strip()
            s2 = self._vars[name+"_high"].get().strip()
            if s1 in ("", "-", ".", "-.") or s2 in ("", "-", ".", "-."):
                raise ValueError
            return (float(s1), float(s2))

        try:
            data["holding_voltage"] = get_float("holding_voltage")
        except Exception:
            errs.append("holding_voltage")

        try:
            data["sampling_interval"] = get_float("sampling_interval")
        except Exception:
            errs.append("sampling_interval")

        try:
            data["peak_search_window_s"] = get_float("peak_search_window_s")
        except Exception:
            errs.append("peak_search_window_s")

        try:
            data["std_factor"] = get_float("std_factor")
        except Exception:
            errs.append("std_factor")

        try:
            data["derivative_smooth_n"] = get_int("derivative_smooth_n")
        except Exception:
            errs.append("derivative_smooth_n")

        try:
            data["min_derivative_neg"] = get_float("min_derivative_neg")
        except Exception:
            errs.append("min_derivative_neg")

        # rated_fit_order ignoriert Eingabe (disabled) – setzen wir auf 1
        data["rated_fit_order"] = 1

        try:
            data["unload_fit_order"] = get_int("unload_fit_order")
        except Exception:
            errs.append("unload_fit_order")

        try:
            data["unloading_low_high"] = get_pair("unloading_low_high")
        except Exception:
            errs.append("unloading_low_high")

        try:
            data["cutaway"] = get_float("cutaway")
        except Exception:
            errs.append("cutaway")

        try:
            data["peak_mean_window_s"] = get_pair("peak_mean_window_s")
        except Exception:
            errs.append("peak_mean_window_s")

        if errs:
            messagebox.showerror("Ungültige Eingaben", "Bitte prüfen: " + ", ".join(errs), parent=self)
            return None

        params = _sanitize(_params_from_dict(data))
        return params

    def _on_save(self):
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

# ========= Quick‑Test: nur Parameter‑Fenster =========
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    params = load_params_from_yaml(DEFAULT_YAML_PATH)  # erstellt YAML bei Bedarf automatisch
    dlg = ParamsEditor(root, params, yaml_path=DEFAULT_YAML_PATH)
    dlg.wait_window()
    if dlg.result is not None:
        save_params_to_yaml(dlg.result, DEFAULT_YAML_PATH)