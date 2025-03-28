import os
import subprocess
from logger_config import get_logger

logger = get_logger("SETUP")

# 1. Venv erstellen
venv_path = "fxYogaVenv"
if not os.path.exists(venv_path):
    logger.info("Erstelle virtuelle Umgebung...")
    subprocess.run(["python", "-m", "venv", venv_path])
else:
    logger.info("Virtuelle Umgebung existiert bereits.")

# 2. Requirements installieren
logger.info("Installiere Abhängigkeiten aus requirements.txt...")
subprocess.run([f"{venv_path}/Scripts/pip", "install", "-r", "requirements.txt"])

# 3. Wichtige Dateien prüfen
if not os.path.exists("Messparameter.xlsx"):
    logger.info("Generiere 'Messparameter.xlsx'...")
    import gen_excel  # erzeugt Datei

# 4. Optional ein Testlauf
logger.info("Starte GUI-Test...")
subprocess.run([f"{venv_path}/Scripts/python", "excel_gui.py"])
