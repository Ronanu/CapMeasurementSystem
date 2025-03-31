# log.py
import sys
import logging
from logger_config import get_logger
from logging.handlers import RotatingFileHandler

# 🔧 Einheitlicher Projekt-Logger
logger = get_logger(name="CapSystem", level=logging.DEBUG)

# 📁 Optional: Logfile mit Rotation (max 1 MB, 3 Backups)
file_handler = RotatingFileHandler("system.log", maxBytes=1_000_000, backupCount=3)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)  # Datei bekommt ALLES
logger.addHandler(file_handler)

# 🧯 Globaler Fehler-Handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unbehandelte Ausnahme", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# 🔍 Optional: Beispiel-Filter für Module aktivieren
class ModuleFilter(logging.Filter):
    def __init__(self, allowed_modules):
        super().__init__()
        self.allowed_modules = allowed_modules

    def filter(self, record):
        return record.module in self.allowed_modules

# Beispiel: Nur Logs aus bestimmten Modulen anzeigen
# logger.addFilter(ModuleFilter(["cap_calculations", "signal_transformations"]))

# 🌐 Zugriff auf Log-Queue für GUI-Module (falls benötigt)
def get_recent_logs(n=10):
    if logger.handlers:
        return list(logger.handlers[0].log_queue)[-n:]  # Nur letzte n Logs
    return []

# Test
if __name__ == "__main__":
    logger.debug("Debug-Test")
    logger.info("Info-Test")
    logger.warning("Warnung-Test")
    logger.error("Fehler-Test")
    logger.critical("Kritisch-Test")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Division durch Null")

    print("--- Letzte Logs ---")
    for line in get_recent_logs():
        print(line)
