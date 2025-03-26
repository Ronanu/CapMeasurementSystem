import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from signal_transformations import SignalData

class PeakDetectionProcessor:
    def __init__(self, signal_data: SignalData, ref_signal_data: SignalData, fs=100, cutoff=1.0, rated_time=0, sigma_threshold=2):
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
        self.rated_time = rated_time

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
        anz_valids = 1000
        valid_offset = 50

        for i, value in enumerate(self.signal_data.data["value"]):
            if len(valid_values) < anz_valids:
                valid_values.append(value)
                continue

            mean_value = np.mean(valid_values[0:-valid_offset])
            threshold = self.sigma_threshold * self.std_dev

            if (mean_value - value) >= threshold:
                self.outliers.append((i, value))
            
            # Fenster immer aktualisieren, egal ob Ausreißer oder nicht
            valid_values.append(value)
            valid_values.pop(0)

        # der peak wird detektiert, wenn wenn 100 outliners hintereinander sind

        print("number of outliers: ", len(self.outliers))

###########
        # Schritt 1: Indizes aus self.outliers extrahieren
        outlier_indices = [idx for idx, _ in self.outliers]

        # Schritt 2: Serien von aufeinanderfolgenden Indizes identifizieren
        series_list = []
        current_series = []

        for idx in outlier_indices:
            if not current_series:
                current_series = [idx]
            elif idx == current_series[-1] + 1:
                current_series.append(idx)
            else:
                series_list.append(current_series)
                current_series = [idx]

        # Vergiss nicht, die letzte Serie hinzuzufügen
        if current_series:
            series_list.append(current_series)

        # Schritt 3: Serien nach Länge absteigend sortieren und kürzere abschneiden
        min_series_factor = 10
        series_list.sort(key=lambda s: len(s), reverse=True)

        if series_list:
            max_len = len(series_list[0])
            min_len = max_len / min_series_factor
            for i, s in enumerate(series_list):
                if len(s) < min_len:
                    series_list = series_list[:i]
                    break

        # Schritt 4: Längste Serie verwenden
        if not series_list:
            print("Kein Peak gefunden – keine zusammenhängende Serie von Ausreißern erkannt.")
            peak_index = 0

        # Schritt 5: Top 3 Serien – Zeitpunkte ausgeben
        print("Ausreißer-Serien starten bei:")
        for s in series_list[:20]:
            start_index = s[0]
            start_time = self.signal_data.data['time'][start_index]
            print(f"t = {start_time:.2f} s, länge = {len(s)}")

        
        # Schritt 6: Wähle die Serie, deren Startzeitpunkt der rated_time am nächsten kommt

        best_series = None
        best_start_index = None
        best_start_time = None
        smallest_distance = float('inf')

        for series in series_list:
            start_index = series[0]
            start_time = self.signal_data.data["time"][start_index]
            distance = abs(start_time - self.rated_time)

            if distance < smallest_distance:
                smallest_distance = distance
                best_series = series
                best_start_index = start_index
                best_start_time = start_time

        if best_series is not None:
            peak_index = best_start_index - 1  # eine Stelle vor dem Ausreißerbeginn
            peak_time = self.signal_data.data["time"][peak_index]
            print(f"Serie mit geringster Abweichung zu rated_time = {self.rated_time:.3f} s:")
            print(f"→ Serienstart: t = {best_start_time:.3f} s (Index {best_start_index})")
            print(f"→ Gewählter Peak: t = {peak_time:.3f} s (Index {peak_index})")
            print(f"→ Abweichung: {smallest_distance:.6f} s")
        else:
            print("Keine gültige Serie gefunden.")
            peak_index = 0

        print(f"Peak gewählt bei Index {peak_index}")
        print(f"Startzeitpunkt der Serie: {best_start_time:.2f} s")
        print(f"Zielzeitpunkt (rated_time): {self.rated_time:.2f} s")
        # print(f"Abweichung: {best_distance:.4f} s")
    

###########
        
        # peak_index = self.outliers[0][0] - 1
        peak_time = self.signal_data.data["time"][peak_index]
        peak_value = self.signal_data.data["value"][peak_index]
        peak_window = self.signal_data.data["value"][peak_index - 10: peak_index]
        peak_mean = np.mean(peak_window)
        
        self.peak = {"time": peak_time, "value": peak_value, "mean": peak_mean, "threshold": threshold}
        return peak_index, peak_time, peak_value, peak_mean, threshold

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