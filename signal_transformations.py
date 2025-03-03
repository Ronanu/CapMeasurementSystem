
from pandas import read_csv, DataFrame
from csv import writer as csv_writer
from numpy import gradient, arange, convolve, ones, polyfit
from matplotlib.pyplot import subplots, tight_layout, show
from scipy.signal import medfilt




class SignalDataLoader:
    def __init__(self, file_path, name, sampling_interval=0.01):
        """
        Lädt eine CSV-Datei und erstellt ein SignalData-Objekt.
        :param file_path: Pfad zur CSV-Datei
        :param name: Name des Signals
        :param sampling_interval: Abtastintervall in Sekunden
        """
        self.file_path = file_path
        self.sampling_interval = sampling_interval
        self.signal_data = self.load_data(name)

    def load_data(self, name):
        """Lädt die CSV-Datei und gibt ein SignalData-Objekt zurück."""
        df = read_csv(self.file_path, delimiter=",", decimal=",", quotechar='"',
                         skipinitialspace=True, usecols=[1], names=["value"], skiprows=1)
        df["value"] = df["value"].astype(str).str.replace(",", ".").astype(float)
        df.dropna(subset=["value"], inplace=True)
        df["time"] = arange(len(df)) * self.sampling_interval
        return SignalData(name, df["time"], df["value"])
    

class SignalData:
    def __init__(self, name, time, values):
        """
        Repräsentiert ein Signal mit Zeit- und Werte-Daten sowie der ersten Ableitung.
        :param name: Name des Signals
        :param time: Zeitwerte als Liste oder NumPy-Array
        :param values: Spannungs- oder Stromwerte als Liste oder NumPy-Array
        """
        self.name = name
        self.data = DataFrame({"time": time, "value": values})
        self.start_time_diff = 0
        self.data["derivative"] = None
        
    def get_derivative(self):
        """Berechnet die erste Ableitung des Signals."""
        self.data["derivative"] = gradient(self.data["value"], self.data["time"])
        return self.data["derivative"]
    
    def get_derivative_signal(self):
        return SignalData(self.name + " (Derivative)", self.data["time"], self.data["derivative"])
    
    def get_data(self):
        """Gibt das DataFrame zurück."""
        return self.data.copy()
    
    # Methode, die Zeitdauer zurückgibt
    def get_time_duration(self):
        return self.data["time"][-1] - self.data["time"][0]
    
    def get_start_and_end_time(self):
        if self.data.empty:
            raise ValueError("Das Signal enthält keine Datenpunkte nach dem Schneiden.")
        return self.data["time"].iloc[0], self.data["time"].iloc[-1]

    
    def set_start_time(self, start_time):   
        # setzt die Startzeit auf den übergebenen Wert und passt die Zeitwerte entsprechend an
        if not self.start_time_diff==0:
            raise ValueError("Startzeit wurde bereits verschoben. Bitte setzen Sie die Startzeit vorher zurück.")
        self.start_time_diff = self.data["time"][0] + start_time
        self.data["time"] = self.data["time"] - self.data["time"][0] + start_time

    def reset_start_time(self):
        starttime, _ = self.get_start_and_end_time()
        self.start_time_diff = 0
        self.set_start_time(-starttime+self.start_time_diff)
    
    # giebt die anzahl der datenpunkte zurück
    def get_number_of_data_points(self):
        return len(self.data["time"])
    
    
    # Überladung der Grundrechenarten mit Skalaren
    def __add__(self, scalar):
        if isinstance(scalar, (int, float)):
            return SignalData(f"{self.name} + {scalar}", self.data["time"], self.data["value"] + scalar)
        raise TypeError("Addition ist nur mit Skalarwerten (int, float) möglich.")

    def __sub__(self, scalar):
        if isinstance(scalar, (int, float)):
            return SignalData(f"{self.name} - {scalar}", self.data["time"], self.data["value"] - scalar)
        raise TypeError("Subtraktion ist nur mit Skalarwerten (int, float) möglich.")

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return SignalData(f"{self.name} * {scalar}", self.data["time"], self.data["value"] * scalar)
        raise TypeError("Multiplikation ist nur mit Skalarwerten (int, float) möglich.")

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError("Division durch null ist nicht erlaubt.")
            return SignalData(f"{self.name} / {scalar}", self.data["time"], self.data["value"] / scalar)
        raise TypeError("Division ist nur mit Skalarwerten (int, float) möglich.")

class SignalDataSaver:
    def __init__(self, signal_data: SignalData, filename="signal_data.csv", header_info=None):
        """
        Initialisiert die Klasse mit einem SignalData-Objekt, einem Dateinamen und optionalen Header-Informationen.
        :param signal_data: SignalData-Objekt mit Zeit- und Werte-Daten
        :param filename: Name der CSV-Datei
        :param header_info: Dictionary mit zusätzlichen Header-Informationen
        """
        self.signal_data = signal_data
        self.filename = filename
        self.header_info = header_info if isinstance(header_info, dict) else {}

    def save_to_csv(self):
        """
        Speichert das SignalData-Objekt als CSV mit 10 Zeilen Header-Informationen.
        """
        if self.signal_data.data.empty:
            print("Keine Signal-Daten zum Speichern.")
            return

        with open(self.filename, mode="w", newline="") as file:
            writer = csv_writer(file)

            # Header in den ersten 10 Zeilen abspeichern
            writer.writerow(["Signal Name", self.signal_data.name])  # Name des Signals in die erste Zeile
            for key, value in self.header_info.items():
                writer.writerow([key, value])  # Speichert Schlüssel-Wert-Paare
            
            # Falls weniger als 10 Header-Zeilen, füllen wir die restlichen mit Leerzeilen
            for _ in range(10 - (len(self.header_info) + 1)):  
                writer.writerow([])

            # Speichert das DataFrame mit Spaltennamen
            self.signal_data.data.to_csv(file, index=False)

        print(f"SignalData erfolgreich in {self.filename} gespeichert.")


class SignalCutter:
    def __init__(self, signal_data):
        """
        Klasse zum Beschneiden eines Signals basierend auf Zeit- oder Wertgrenzen.
        :param signal_data: Ein SignalData-Objekt
        """
        self.original_data = signal_data
    
    def cut_time_range(self, time_range):
        """Beschneidet das Signal im angegebenen Zeitbereich."""
        df = self.original_data.get_data()
        df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])].reset_index(drop=True)
        return SignalData(self.original_data.name + " (Time Cut)", df["time"], df["value"])
    
    def cut_by_value(self, direction, threshold):
        """
        Bsp.: cut_by_value("l>", 1) entfernt alle Werte von links, bis der erste Wert größer als 1 ist.
        :param direction: "l>" (links größer), "l<" (links kleiner), "r>" (rechts größer), "r<" (rechts kleiner)
        :param threshold: Schwellenwert für die Beschneidung
        """
        df = self.original_data.get_data()
        
        if direction == "l>":
            first_idx = df[df["value"] > threshold].index.min()
            if first_idx is None:
                raise ValueError(f"Kein Wert größer als {threshold} gefunden.")
            if first_idx == 0:
                print(f"Warnung: Keine Datenpunkte wurden durch {direction} {threshold} entfernt.")
            df = df.loc[first_idx:].reset_index(drop=True)
            # print direction, last_idx and len(df) to debug
            print(f"{direction} {first_idx} of {len(df)}")
        
        elif direction == "l<":
            first_idx = df[df["value"] < threshold].index.min()
            if first_idx is None:
                raise ValueError(f"Kein Wert kleiner als {threshold} gefunden.")
            if first_idx == 0:
                print(f"Warnung: Keine Datenpunkte wurden durch {direction} {threshold} entfernt.")
            df = df.loc[first_idx:].reset_index(drop=True)
            # print direction, last_idx and len(df) to debug
            print(f"{direction} {first_idx} of {len(df)}")
        
        elif direction == "r>":
            last_idx = df[df["value"] > threshold].index.max()
            if last_idx is None:
                raise ValueError(f"Kein Wert größer als {threshold} gefunden.")
            if last_idx == len(df) - 1:
                print(f"Warnung: Keine Datenpunkte wurden durch {direction} {threshold} entfernt.")
            df = df.loc[:last_idx].reset_index(drop=True)
            # print direction, last_idx and len(df) to debug
            print(f"{direction} {last_idx} of {len(df)}")
        
        elif direction == "r<":
            last_idx = df[df["value"] < threshold].index.max()
            if last_idx is None:
                raise ValueError(f"Kein Wert kleiner als {threshold} gefunden.")
            if last_idx == len(df) - 1:
                print(f"Warnung: Keine Datenpunkte wurden durch {direction} {threshold} entfernt.")
            df = df.loc[:last_idx].reset_index(drop=True)
            # print direction, last_idx and len(df) to debug
            print(f"{direction} {last_idx} of {len(df)}")
        
        else:
            raise ValueError("Ungültige Richtung. Verwenden Sie 'l>', 'l<', 'r>' oder 'r<'.")
        
        return SignalData(
            self.original_data.name + f" (Cut {direction} {threshold})",
            df["time"],
            df["value"]
        )


class MedianFilter:
    def __init__(self, signal_data, window_size=5):
        """
        Wendet einen Medianfilter auf ein SignalData-Objekt an.
        :param signal_data: Ein SignalData-Objekt
        :param window_size: Fenstergröße für den Medianfilter
        """
        self.original_data = signal_data
        self.signal_data = self.apply_filter(window_size)
    
    def apply_filter(self, window_size):
        """Berechnet das gefilterte Signal und gibt ein neues SignalData-Objekt zurück."""
        df = self.original_data.get_data()
        df["value"] = medfilt(df["value"], kernel_size=window_size)
        return SignalData(self.original_data.name + " (Median)", df["time"], df["value"])
    

class MovingAverageFilter:
    def __init__(self, signal_data, window_size=5):
        """
        Wendet einen gleitenden Mittelwertfilter auf ein SignalData-Objekt an.
        :param signal_data: Ein SignalData-Objekt
        :param window_size: Fenstergröße für den Mittelwertfilter
        """
        self.original_data = signal_data
        self.signal_data = self.apply_filter(window_size)
    
    def apply_filter(self, window_size):
        """Berechnet das gefilterte Signal und gibt ein neues SignalData-Objekt zurück."""
        df = self.original_data.get_data()
        df["value"] = df["value"].rolling(window=window_size, center=True).mean().bfill().ffill()
        return SignalData(self.original_data.name + " (Moving Average)", df["time"], df["value"])

class ConvolutionSmoothingFilter:
    def __init__(self, signal_data, kernel_size=5):
        """
        Wendet eine Glättung mittels Faltung auf ein SignalData-Objekt an.
        :param signal_data: Ein SignalData-Objekt
        :param kernel_size: Größe des Glättungskerns
        """
        self.original_data = signal_data
        self.signal_data = self.apply_filter(kernel_size)
    
    def apply_filter(self, kernel_size):
        """Berechnet das gefilterte Signal durch Faltung und gibt ein neues SignalData-Objekt zurück."""
        df = self.original_data.get_data()
        kernel = ones(kernel_size) / kernel_size
        df["value"] = convolve(df["value"], kernel, mode='same')
        return SignalData(self.original_data.name + " (Convolution Smoothing)", df["time"], df["value"])


class PlotVoltageAndCurrent:
    def __init__(self, voltage_signals, current_signals):
        """
        Erstellt einen Plot für Spannung und Strom.
        :param voltage_signals: Liste von SignalData-Objekten für Subplot 1 (Spannung)
        :param current_signals: Liste von SignalData-Objekten für Subplot 2 (Strom)
        """
        self.voltage_signals = voltage_signals
        self.current_signals = current_signals
        self.plot_signal()

    def plot_signal(self):
        """Erstellt den Plot."""
        fig, axs = subplots(2, 1, figsize=(12, 8), sharex=True)
        
        for signal in self.voltage_signals:
            axs[0].plot(signal.get_data()["time"], signal.get_data()["value"], label=signal.name)
        axs[0].set_ylabel("Spannung (V)")
        axs[0].set_title("Spannungssignale")
        axs[0].legend()
        axs[0].grid()
        
        for signal in self.current_signals:
            axs[1].plot(signal.get_data()["time"], signal.get_data()["value"], label=signal.name)
        axs[1].set_xlabel("Zeit (s)")
        axs[1].set_ylabel("Strom (A)")
        axs[1].set_title("Stromsignale")
        axs[1].legend()
        axs[1].grid()
        
        tight_layout()


class CapacitorEvaluation:
    def __init__(self, signal_data, U_apl=3, I_dis=1):
        self.signal_data = signal_data
        self.U_apl = U_apl
        self.I_dis = I_dis
        self.results = {}
    
    def lsm_fit(self, order=1):
        coeff = polyfit(self.signal_data.data["time"], self.signal_data.data["value"], order)
        return coeff
    
    def find_point_of_start_discharging(self, U_max, tolerance):
        for i in range(len(self.signal_data.data)):
            if (U_max - self.signal_data.data["value"].iloc[i]) > tolerance:
                index_start_discharge = i - 1
                return self.signal_data.data["time"].iloc[index_start_discharge], self.signal_data.data["value"].iloc[index_start_discharge], index_start_discharge
        print("Fehler! Punkt, wo Entladevorgang startet, wurde nicht gefunden!")
        return None, None, None
    

def testing_signal_cutter():
    file_path = "csv_files/MAL2_5A2esr.csv"
    raw_signal = SignalDataLoader(file_path, "Original Signal").signal_data
    threshold = 1  # Beispiel-Schwellenwert für die Spannung
    signal_cutter = SignalCutter(raw_signal)    
    cut_larger_left = signal_cutter.cut_by_value("l>", threshold) + 0.1
    signal_cutter = SignalCutter(cut_larger_left)
    cut_smaller_left = signal_cutter.cut_by_value("l<", threshold) + 0.2
    signal_cutter = SignalCutter(raw_signal)
    cut_larger_right = signal_cutter.cut_by_value("r>", threshold)
    signal_cutter = SignalCutter(cut_larger_right)
    cut_smaller_right = signal_cutter.cut_by_value("r<", threshold) + 0.1

    PlotVoltageAndCurrent(
    voltage_signals=[cut_larger_left, cut_smaller_left],
    current_signals=[cut_larger_right, cut_smaller_right]
    )

    cut_larger_left.set_start_time(0)
    cut_smaller_left.set_start_time(0)
    cut_larger_right.set_start_time(500)
    cut_smaller_right.set_start_time(500)

    PlotVoltageAndCurrent(
    voltage_signals=[cut_larger_left, cut_smaller_left],
    current_signals=[cut_larger_right, cut_smaller_right]
    )

    cut_larger_left.reset_start_time()
    cut_smaller_left.reset_start_time()
    cut_larger_right.reset_start_time()
    cut_smaller_right.reset_start_time()

    PlotVoltageAndCurrent(
    voltage_signals=[cut_larger_left, cut_smaller_left],
    current_signals=[cut_larger_right, cut_smaller_right]
    )

if __name__ == "__main__":
    testing_signal_cutter()
    show()