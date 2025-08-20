import os
import subprocess
from logger_config import get_logger

logger = get_logger("SETUP")

# 1. Venv erstellen
venv_path = "venv"
if not os.path.exists(venv_path):
    logger.info("Erstelle virtuelle Umgebung...")
    subprocess.run(["python", "-m", "venv", venv_path])
else:
    logger.info("Virtuelle Umgebung existiert bereits.")

# 2. Requirements installieren
logger.info("Installiere Abhängigkeiten aus requirements.txt...")
subprocess.run([f"{venv_path}/Scripts/pip", "install", "-r", "requirements.txt"])

