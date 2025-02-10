import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SignalProcessor:
    def __init__(self, file_path, threshold_voltage=0.5, sampling_interval=0.01):
        """
        Initialisiert den Signalprozessor und lädt die Daten aus der Datei.

        :param file_path: Pfad zur CSV-Datei.
        :param threshold_voltage: Schwelle zur Beschneidung des Signals (V).
        :param sampling_interval: Zeit zwischen zwei Messpunkten (s).
        """
        self.file_path = file_path
        self.threshold_voltage = threshold_voltage
        self.sampling_interval = sampling_interval
        self.samples_per_second = int(1 / sampling_interval)

        self.df = self.load_data()
        self.df_trimmed = self.trim_signal()
        self.df_trimmed["dU/dt"] = self.compute_derivative()  # <-- Korrektur: Speichern in df_trimmed
        self.average_dU_dt = self.compute_segmented_average()

    def load_data(self):
        """Lädt die CSV-Datei ein und wandelt sie in ein DataFrame um."""
        df = pd.read_csv(self.file_path, delimiter=",", decimal=",", quotechar='"',
                         skipinitialspace=True, usecols=[1], names=["Spannung (V)"], skiprows=1)

        # Konvertiere die Spannungswerte in numerische Werte
        df["Spannung (V)"] = df["Spannung (V)"].astype(str).str.replace(",", ".").astype(float)

        # Füge eine Zeitspalte hinzu (10 ms Abtastzeit)
        df["Zeit (s)"] = np.arange(len(df)) * self.sampling_interval
        return df

    def trim_signal(self):
        """Beschneidet das Signal basierend auf der Schwelle threshold_voltage."""
        valid_indices = self.df[self.df["Spannung (V)"] >= self.threshold_voltage].index

        if not valid_indices.empty:
            return self.df.loc[valid_indices.min():valid_indices.max()].reset_index(drop=True)
        else:
            return self.df.copy()

    def compute_derivative(self):
        """Berechnet die erste Ableitung dU/dt."""
        return np.gradient(self.df_trimmed["Spannung (V)"], self.sampling_interval)

    def compute_segmented_average(self):
        """Berechnet die mittlere Ableitung pro Sekunde."""
        num_full_seconds = len(self.df_trimmed) // self.samples_per_second
        mean_derivatives = []

        for i in range(num_full_seconds):
            start_idx = i * self.samples_per_second
            end_idx = start_idx + self.samples_per_second
            segment = self.df_trimmed.iloc[start_idx:end_idx]
            mean_derivatives.append(segment["dU/dt"].mean())

        # Wiederholung der Mittelwerte für alle Punkte
        diff = len(self.df_trimmed) - len(mean_derivatives) * self.samples_per_second
        if diff > 0:
            repeated_values = np.append(np.repeat(mean_derivatives, self.samples_per_second), 
                                        [mean_derivatives[-1]] * diff)
        elif diff < 0:
            repeated_values = np.repeat(mean_derivatives, self.samples_per_second)[:len(self.df_trimmed)]
        else:
            repeated_values = np.repeat(mean_derivatives, self.samples_per_second)

        # Sicherstellen, dass die Länge übereinstimmt
        assert len(repeated_values) == len(self.df_trimmed), "Fehlanpassung der Längen"
        return repeated_values

    def plot_signal(self):
        """Erstellt einen doppelten Subplot mit dem Originalsignal und den Ableitungen."""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Oberes Diagramm: Originalsignal
        axs[0].plot(self.df_trimmed["Zeit (s)"], self.df_trimmed["Spannung (V)"], color="blue", label="Spannung (V)")
        axs[0].set_ylabel("Spannung (V)")
        axs[0].set_title("Originalsignal (beschnitten)")
        axs[0].legend()
        axs[0].grid()

        # Unteres Diagramm: Ableitung (gesamte Ableitung vs. segmentierte Mittelwerte)
        axs[1].plot(self.df_trimmed["Zeit (s)"], self.df_trimmed["dU/dt"], color="gray", alpha=0.5, label="Gesamte Ableitung (dU/dt)")
        axs[1].plot(self.df_trimmed["Zeit (s)"], self.average_dU_dt, color="red", label="Segmentierte mittlere Ableitung")
        axs[1].set_xlabel("Zeit (s)")
        axs[1].set_ylabel("Ableitung (dU/dt)")
        axs[1].set_title("Ableitung: Rohdaten vs. segmentierter Durchschnitt")
        axs[1].axhline(0, color="black", linestyle="dashed", linewidth=1)
        axs[1].legend()
        axs[1].grid()

        # Anzeige des Plots
        plt.tight_layout()
        plt.show()

# Instanziiere die Klasse und erstelle den Plot
file_path = "MAL2_6A2.csv"
processor = SignalProcessor(file_path)
processor.plot_signal()