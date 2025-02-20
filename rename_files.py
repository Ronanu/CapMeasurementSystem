import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog

def parse_filename(fname, file_extension=".picolog"):
    """
    Zerlegt den Dateinamen in die Eigenschaften:
      Cap_nr, Method, Special
    und gibt außerdem den neuen Dateinamen zurück.
    """
    # Dateiendung entfernen
    name = re.sub(rf"\{file_extension}$", "", fname)

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
        new_name = f"cap{cap_nr}_{method}_{special}{file_extension}"
    else:
        new_name = f"cap{cap_nr}_{method}{file_extension}"

    return cap_nr, method, special, new_name

def process_files(file_extension=".picolog"):
    """ Öffnet einen Dialog zur Auswahl des Quellordners und verarbeitet die Dateien """
    # GUI-Fenster für Ordnerauswahl
    root = tk.Tk()
    root.withdraw()  # Fenster verstecken
    src_folder = filedialog.askdirectory(title="Wähle den Quellordner mit den Dateien")

    if not src_folder:  # Falls kein Ordner gewählt wurde, abbrechen
        print("Kein Ordner ausgewählt. Vorgang abgebrochen.")
        return

    # Dateiendung abfragen (mit Vorgabe aus Parameter)
    user_extension = simpledialog.askstring("Dateiendung", 
                                            "Gib die gewünschte Dateiendung ein (Standard: .picolog):",
                                            initialvalue=file_extension)
    
    if not user_extension:
        user_extension = file_extension  # Falls keine Eingabe, Standardwert nutzen
    
    if not user_extension.startswith("."):
        user_extension = "." + user_extension  # Sicherstellen, dass die Endung mit einem Punkt beginnt

    dst_folder = os.path.join(src_folder, "processed_files")
    os.makedirs(dst_folder, exist_ok=True)

    # Alle Dateien mit der gewählten Endung im Quellordner finden
    filenames = [f for f in os.listdir(src_folder) if f.endswith(user_extension)]

    if not filenames:
        print(f"Keine Dateien mit der Endung '{user_extension}' gefunden.")
        return

    for fname in filenames:
        old_path = os.path.join(src_folder, fname)

        # Eigenschaften extrahieren und neuen Namen generieren
        cap_nr, method, special, new_name = parse_filename(fname, user_extension)

        # Zielpfad mit neuem Namen
        new_path = os.path.join(dst_folder, new_name)

        # Kopieren mit neuem Namen
        shutil.copy2(old_path, new_path)
        print(f"Kopiert: {fname} → {new_name}")

    print(f"Alle Dateien wurden in '{dst_folder}' gespeichert.")

# Starte das Skript
if __name__ == "__main__":
    process_files()
