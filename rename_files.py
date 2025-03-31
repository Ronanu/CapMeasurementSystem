import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog
from log import logger

def parse_filename(fname, file_extension=".picolog"):
    name = re.sub(rf"\{file_extension}$", "", fname)
    name = re.sub(r"^MAL2_", "", name)

    cap_nr = None
    method = None
    special = None

    if name.startswith("np"):
        special = "after_press"
        name = name[2:]
    elif name.startswith("p"):
        special = "pressed"
        name = name[1:]

    if "A2esr" in name:
        method = "ESR_A2_Class2"
        name = name.replace("A2esr", "")
    elif "A2" in name:
        method = "C_A1_Class2"
        name = name.replace("A2", "")
    else:
        method = "C_B1"

    match_nr = re.search(r"(\d+)", name)
    cap_nr = match_nr.group(1) if match_nr else "unknown"

    if special is not None and 'ress' in special:
        if cap_nr == 1:
            cap_nr = 5
        elif cap_nr == 2:
            cap_nr = 3
        elif cap_nr == 3:
            cap_nr = 8

    hersteller = 'Vishay'
    Cn = 50

    if special is None:
        new_name = f"{method}_DUT{cap_nr}_V1_{hersteller}_{Cn}{file_extension}"
    else:
        new_name = f"{method}_DUT{cap_nr}_V1_{hersteller}_{special}_{Cn}{file_extension}"

    logger.debug(
        f"Dateiname '{fname}' analysiert → "
        f"CapNr: {cap_nr}, Method: {method}, Special: {special}, Neuer Name: {new_name}"
    )

    return cap_nr, method, special, new_name

def process_files(file_extension=".picolog"):
    root = tk.Tk()
    root.withdraw()
    src_folder = filedialog.askdirectory(title="Wähle den Quellordner mit den Dateien")

    if not src_folder:
        logger.warning("Kein Ordner ausgewählt. Vorgang abgebrochen.")
        return

    user_extension = simpledialog.askstring("Dateiendung",
                                            "Gib die gewünschte Dateiendung ein (Standard: .picolog):",
                                            initialvalue=file_extension)

    if not user_extension:
        user_extension = file_extension

    if not user_extension.startswith("."):
        user_extension = "." + user_extension

    dst_folder = os.path.join(src_folder, "processed_files")
    os.makedirs(dst_folder, exist_ok=True)

    filenames = [f for f in os.listdir(src_folder) if f.endswith(user_extension)]

    if not filenames:
        logger.info(f"Keine Dateien mit der Endung '{user_extension}' gefunden.")
        return

    logger.info(f"{len(filenames)} Dateien mit Endung '{user_extension}' gefunden. Beginne Verarbeitung …")

    for fname in filenames:
        old_path = os.path.join(src_folder, fname)

        try:
            cap_nr, method, special, new_name = parse_filename(fname, user_extension)
            new_path = os.path.join(dst_folder, new_name)
            shutil.copy2(old_path, new_path)
            logger.debug(f"Datei kopiert: '{fname}' → '{new_name}'")
        except Exception as e:
            logger.error(f"Fehler bei Verarbeitung von '{fname}': {e}")

    logger.info(f"Verarbeitung abgeschlossen. Dateien gespeichert in: {dst_folder}")

if __name__ == "__main__":
    process_files()
