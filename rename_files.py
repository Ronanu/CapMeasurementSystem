import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog

def parse_filename(fname):
    """
    Zerlegt den Dateinamen in die Eigenschaften:
      Cap_nr, Method, Special
    und gibt außerdem den neuen Dateinamen zurück.
    """
    # ".picolog" am Ende entfernen
    name = re.sub(r"\.picolog$", "", fname)

    # "MAL2_" entfernen, falls vorhanden
    name = re.sub(r"^MAL2_", "", name)

    # Eigenschaften initialisieren
    cap_nr = None
    method = None
    special = None

    # Spezialfälle prüfen: "np" oder "p" am Anfang
    if name.startswith("np"):
        special = "nach_Pressung"
        name = name[2:]  # "np" entfernen
    elif name.startswith("p"):
        special = "pressed"
        name = name[1:]  # "p" entfernen

    # Methode bestimmen
    if "A2esr" in name:
        method = "A2esr"
        name = name.replace("A2esr", "")
    elif "A2" in name:
        method = "A2"
        name = name.replace("A2", "")
    else:
        method = "B2"

    # Koppelkondensator-Nummer extrahieren
    match_nr = re.search(r"(\d+)", name)
    cap_nr = match_nr.group(1) if match_nr else "unknown"

    # Neuen Dateinamen erstellen
    if special:
        new_name = f"cap{cap_nr}_{method}_{special}.picolog"
    else:
        new_name = f"cap{cap_nr}_{method}.picolog"

    return cap_nr, method, special, new_name

def process_files():
    """ Öffnet einen Dialog zur Auswahl des Quellordners und verarbeitet die Dateien """
    # GUI-Fenster für Ordnerauswahl
    root = tk.Tk()
    root.withdraw()  # Fenster verstecken
    src_folder = filedialog.askdirectory(title="Wähle den Quellordner mit .picolog-Dateien")

    if not src_folder:  # Falls kein Ordner gewählt wurde, abbrechen
        print("Kein Ordner ausgewählt. Vorgang abgebrochen.")
        return

    dst_folder = os.path.join(src_folder, "processed_files")
    os.makedirs(dst_folder, exist_ok=True)

    # Alle .picolog-Dateien im Quellordner finden
    filenames = [f for f in os.listdir(src_folder) if f.endswith(".picolog")]

    for fname in filenames:
        old_path = os.path.join(src_folder, fname)

        # Eigenschaften extrahieren und neuen Namen generieren
        cap_nr, method, special, new_name = parse_filename(fname)

        # Zielpfad mit neuem Namen
        new_path = os.path.join(dst_folder, new_name)

        # Kopieren mit neuem Namen
        shutil.copy2(old_path, new_path)
        print(f"Kopiert: {fname} → {new_name}")

    print(f"Alle Dateien wurden in '{dst_folder}' gespeichert.")

# Starte das Skript
if __name__ == "__main__":
    process_files()
