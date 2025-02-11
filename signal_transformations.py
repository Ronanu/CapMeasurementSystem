import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

class SignalData:
    def __init__(self, name, time, values):
        """
        Repräsentiert ein Signal mit Zeit- und Werte-Daten sowie der ersten Ableitung.
        :param name: Name des Signals
        :param time: Zeitwerte als Liste oder NumPy-Array
        :param values: Spannungs- oder Stromwerte als Liste oder NumPy-Array
        """
        self.name = name
        self.data = pd.DataFrame({"time": time, "value": values})
        self.data["derivative"] = self.get_derivative()

    def get_derivative(self):
        """Berechnet die erste Ableitung des Signals."""
        return np.gradient(self.data["value"], self.data["time"])
    
    def get_derivative_signal(self):
        return SignalData(self.name + " (Derivative)", self.data["time"], self.data["derivative"])
    
    def get_data(self):
        """Gibt das DataFrame zurück."""
        return self.data.copy()

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
        df = pd.read_csv(self.file_path, delimiter=",", decimal=",", quotechar='"',
                         skipinitialspace=True, usecols=[1], names=["value"], skiprows=1)
        df["value"] = df["value"].astype(str).str.replace(",", ".").astype(float)
        df["time"] = np.arange(len(df)) * self.sampling_interval
        return SignalData(name, df["time"], df["value"])
    
    def get_derivative_signal(self):
        """Gibt ein neues SignalData-Objekt zurück, das die Ableitung enthält."""
        df = self.signal_data.get_data()
        return SignalData(self.signal_data.name + " (Derivative)", df["time"], df["derivative"])

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
        df = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1])]
        return SignalData(self.original_data.name + " (Time Cut)", df["time"], df["value"])
    
    def cut_voltage_range(self, direction, threshold):
        """Beschneidet das Signal basierend auf Spannungswerten.
        :param direction: "l>" (links größer), "l<" (links kleiner), "r>" (rechts größer), "r<" (rechts kleiner)
        :param threshold: Schwellenwert für die Beschneidung
        """
        df = self.original_data.get_data()
        if direction == "l>":
            first_idx = df[df["value"] > threshold].index.min()
            df = df.loc[first_idx:]
        elif direction == "l<":
            first_idx = df[df["value"] < threshold].index.min()
            df = df.loc[first_idx:]
        elif direction == "r>":
            last_idx = df[df["value"] > threshold].index.max()
            df = df.loc[:last_idx]
        elif direction == "r<":
            last_idx = df[df["value"] < threshold].index.max()
            df = df.loc[:last_idx]
        return SignalData(self.original_data.name + " (Voltage Cut)", df["time"], df["value"])

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
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
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
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = "MAL2_6A2.csv"
    raw_signal = SignalDataLoader(file_path, "Original Signal").signal_data
    cut_signal = SignalCutter(raw_signal).cut_time_range((1, 200))
    filtered_signal = MedianFilter(cut_signal, window_size=7).signal_data
    derived_signal = raw_signal.get_derivative_signal()
    PlotVoltageAndCurrent(voltage_signals=[cut_signal, filtered_signal], current_signals=[derived_signal])
