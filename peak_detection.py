import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from signal_transformations import SignalData

class PeakDetectionProcessor:
    def __init__(self, signal_data: SignalData, ref_signal_data: SignalData, fs=100, cutoff=1.0, sigma_threshold=2):
        """
        Initialisiert die Klasse für die Peak-Erkennung.
        :param signal_data: Signal, an dem die Peak-Detection ausgeführt wird
        :param ref_signal_data: Signal zur Berechnung der Standardabweichung
        :param fs: Abtastrate in Hz
        :param cutoff: Grenzfrequenz für Hochpass-Filter in Hz
        :param sigma_threshold: Faktor für die Standardabweichung zur Ausreißer-Erkennung
        """
        self.signal_data = signal_data
        self.ref_signal_data = ref_signal_data
        self.fs = fs
        self.cutoff = cutoff
        self.sigma_threshold = sigma_threshold
        self.filtered_data = None
        self.std_dev = None
        self.outliers = []
        self.peak = {}

    def high_pass_filter(self):
        """Wendet einen Butterworth-Hochpassfilter an, um Gleichspannung zu entfernen."""
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_values = signal.filtfilt(b, a, self.signal_data.data["value"])
        self.filtered_data = SignalData(self.signal_data.name + " (HP-Filtered)",
                                        self.signal_data.data["time"], filtered_values)

    def compute_standard_deviation(self):
        """Berechnet die Standardabweichung aus dem separaten Referenzsignal."""
        self.std_dev = np.std(self.ref_signal_data.data["value"])

    def detect_peaks(self):
        """Erkennt Peaks anhand der 2-Sigma-Regel."""
        if self.std_dev is None:
            raise ValueError("Standardabweichung muss zuerst berechnet werden!")

        valid_values = []
        for i, value in enumerate(self.signal_data.data["value"]):
            if len(valid_values) < 10:  # Erste 10 Werte als Basis für Mittelwert
                valid_values.append(value)
                continue
            
            mean_value = np.mean(valid_values)
            if abs(value - mean_value) > self.sigma_threshold * self.std_dev:
                self.outliers.append((i, value))  # Index und Wert speichern
            else:
                valid_values.append(value)
                valid_values.pop(0)
        
        peak_index = self.outliers[0][0] - 1
        peak_time = self.signal_data.data["time"][peak_index]
        peak_value = self.signal_data.data["value"][peak_index]
        peak_window = self.signal_data.data["value"][peak_index - 10: peak_index]
        peak_mean = np.mean(peak_window)
        threshold = self.sigma_threshold * self.std_dev
        self.peak = {"time": peak_time, "value": peak_value, "mean": peak_mean, "threshold": threshold}
        return peak_time, peak_value, peak_mean, threshold

    def plot_results(self):
        """Visualisiert die Daten, gefilterten Werte und markiert Peaks."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.signal_data.data["time"], self.signal_data.data["value"], label="Originaldaten", alpha=0.5)
        plt.plot(self.filtered_data.data["time"], self.filtered_data.data["value"], label="Hochpass-gefiltert", linewidth=2)

        if self.outliers:
            outlier_indices, outlier_values = zip(*self.outliers)
            plt.scatter(self.signal_data.data["time"][list(outlier_indices)], outlier_values, color='red', label="Peaks", zorder=3)

        # Markierung des gefundenen Peaks
        plt.axvline(self.peak["time"], color='green', linestyle='--', label="Peak-Zeitpunkt")
        plt.axhline(self.peak["value"], color='green', linestyle='--', label="Peak-Wert")
        plt.axhline(self.peak["mean"], color='blue', linestyle='--', label="Peak-Mittelwert")
        plt.axhline(self.peak["mean"] + self.peak["threshold"], color='orange', linestyle='--', label="Threshold")
        plt.axhline(self.peak["mean"] - self.peak["threshold"], color='orange', linestyle='--')
        plt.legend()
        plt.xlabel("Zeit (s)")
        plt.ylabel("Signalwert")
        plt.title("Peak-Detection mit Hochpassfilterung")
        

if __name__ == '__main__': 
    # Beispiel-Nutzung:
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    values = np.ones(1000) * 10 + np.random.normal(0, 0.5, 1000)  # Gleichspannung + Rauschen
    values[300] += 5  # Künstlicher Peak
    values[700] -= 4  # Künstlicher Peak

    signal_data = SignalData("Test-Signal", time, values)
    ref_signal_data = SignalData("Referenz-Signal", time[:500], values[:500])  # Erste 500 Werte als Referenz

    processor = PeakDetectionProcessor(signal_data, ref_signal_data, sigma_threshold=4)
    processor.high_pass_filter()
    processor.compute_standard_deviation()
    processor.detect_peaks()
    processor.plot_results()

    print("Gefundene Peaks:", processor.outliers)
    plt.show()