# Quelle: https://github.com/Ronanu/CrazyHouse/blob/dfbccab82a151331f30d228ba7270f7935d6cddb/logger_config.py
# Commit: https://github.com/Ronanu/CrazyHouse/commit/dfbccab82a151331f30d228ba7270f7935d6cddb
# Uebernommen am: 28.03.2025  4:13:35,05

import logging
from collections import deque
from colorama import Fore, Style, init

# Initialize colorama
init()

class CustomHandler(logging.Handler):
    def __init__(self, queue_length=22):
        super().__init__()
        self.log_queue = deque(maxlen=queue_length)

    def emit(self, record):
        color = {
            'DEBUG': Fore.BLUE,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA
        }.get(record.levelname, Fore.WHITE)
        
        log_entry = self.format(record)
        colored_log_entry = f"{color}{log_entry}{Style.RESET_ALL}"
        
        self.log_queue.append(colored_log_entry)
        print(colored_log_entry)

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG or record.levelno >= logging.WARNING:
            record.msg = f"{record.msg} (File: {record.pathname}, Line: {record.lineno}, Function: {record.funcName})"
        return super().format(record)

def get_logger(name=__name__, level=logging.DEBUG, queue_length=22):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    custom_handler = CustomHandler(queue_length=queue_length)
    custom_formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
    custom_handler.setFormatter(custom_formatter)
    logger.addHandler(custom_handler)
    
    return logger

if __name__ == "__main__":
    logger = get_logger(__name__, logging.DEBUG)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")