import keyboard
import pyautogui
import time
import pyperclip
import yaml
import os

class PositionLogger:
    def __init__(self, output_file):
        self.output_file = output_file
        self.actions = []  # Speichert Aktionen als Dicts
        self.start_time = None
        
    def log_position(self):
        x, y = pyautogui.position()
        timestamp = time.time()
        if not self.start_time:
            self.start_time = timestamp
        self.actions.append({
            "type": "click",
            "x": x,
            "y": y,
            "time": timestamp - self.start_time
        })
        self.start_time = timestamp
        print(f"Mausposition gespeichert: ({x}, {y})")
    
    def log_paste(self):
        timestamp = time.time()
        if not self.start_time:
            self.start_time = timestamp
        self.actions.append({
            "type": "paste",
            "time": timestamp - self.start_time
        })
        self.start_time = timestamp
        print("Einfügeaktion (V) gespeichert.")
    
    def log_delay(self):
        timestamp = time.time()
        if not self.start_time:
            self.start_time = timestamp
        self.actions.append({
            "type": "delay",
            "time": timestamp - self.start_time
        })
        self.start_time = timestamp
        print("Eingabe-Wartezeit gespeichert.")
        
    def save_positions(self):
        with open(self.output_file, 'w') as f:
            yaml.dump(self.actions, f)
        print(f"Aktionen gespeichert in {self.output_file}.")
    
    def run(self):
        print("Drücke SPACE für Position, V für Einfügen, ENTER für Wartezeit, ESC zum Beenden und Speichern.")
        while True:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == 'space':
                    self.log_position()
                elif event.name == 'v':
                    self.log_paste()
                elif event.name == 'enter':
                    self.log_delay()
                elif event.name == 'esc':
                    self.save_positions()
                    break

class PositionClicker:
    def __init__(self, input_file):
        self.input_file = input_file
        self.actions = []
        self._load_positions()
    
    def _load_positions(self):
        with open(self.input_file, 'r') as f:
            self.actions = yaml.safe_load(f)
    
    def click_positions(self, v_msg='filename.csv'):
        start_time = time.time()
        for action in self.actions:
            #while time.time() - start_time < action["time"]:
                #time.sleep(0.05)
            if action["type"] == "click":
                pyautogui.click(action["x"], action["y"])
                time.sleep(0.3)
                print(f"Klick an Position: ({action['x']}, {action['y']})")
            elif action["type"] == "paste":
                pyperclip.copy(v_msg)
                time.sleep(0.1)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.3)
                print(f"Dateiname {v_msg} eingefügt.")
            elif action["type"] == "delay":
                print("Warte...")
                time.sleep(action["time"])
                print("Wartezeit durchgeführt.")

class FileProcessor:
    def __init__(self, folder_path, logger_file):
        self.folder_path = folder_path
        self.logger_file = logger_file
    
    def process_files(self):    
        # Liste aller Dateien im Ordner
        all_files = os.listdir(self.folder_path)

        # Filtere nur Dateien mit der Endung '.picolog'
        picolog_files = [f for f in all_files if f.endswith('.picolog')]

        # Sortiere die Dateien nach Größe (absteigend)
        files = sorted(picolog_files, key=lambda f: os.path.getsize(os.path.join(self.folder_path, f)), reverse=True)

        if not files:
            print("Keine Dateien gefunden.")
            return
        
        clicker = PositionClicker(self.logger_file)
        
        for file in files:
            print(f"Öffne Datei: {file}")
            os.startfile(os.path.join(self.folder_path, file))
            time.sleep(5)  # Warte 3 Sekunden
            filename = self.cleanup_filename(file)  # Dateiname bereinigen
            clicker.click_positions(filename)
            time.sleep(2)

    def cleanup_filename(self, filename):
        # filename hat .picolog am ende. ersetze mit csv
        return filename.replace('.picolog', '.csv')

if __name__ == "__main__":
    mode = input("(L)ogger oder (C)licker oder (P)rozess? ").strip().lower()
    if mode == 'l':
        logger = PositionLogger("mouse_positions.yaml")
        logger.run()
    elif mode == 'c':
        clicker = PositionClicker("mouse_positions.yaml")
        clicker.click_positions()
    elif mode == 'p':
        folder = input("Ordner mit Picolog-Dateien: ").strip()
        processor = FileProcessor(folder, "mouse_positions.yaml")
        processor.process_files()
    else:
        print("Ungültiger Modus.")