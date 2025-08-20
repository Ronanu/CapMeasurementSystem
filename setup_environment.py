import os
import subprocess
from logger_config import get_logger

logger = get_logger("SETUP")

venv_path = "venv"

# 1. Virtuelle Umgebung erstellen
if not os.path.exists(venv_path):
    logger.info("Erstelle virtuelle Umgebung...")
    subprocess.run(["python", "-m", "venv", venv_path], check=True)
else:
    logger.info("Virtuelle Umgebung existiert bereits.")

# 2. Abhängigkeiten installieren
pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
if os.path.exists("requirements.txt"):
    logger.info("Installiere Abhängigkeiten aus requirements.txt...")
    subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
else:
    logger.warning("Keine requirements.txt gefunden – überspringe Installation.")
