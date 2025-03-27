from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal
from cap_signal_processing import polynomial_fit, evaluate_polynomial
from signal_transformations import SignalDataLoader, SignalCutter, SignalDataSaver
from peak_detection import PeakDetectionProcessor

from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from os.path import dirname, basename, join, isfile, exists
from numpy import inf
from tkinter import filedialog, Tk
from os import listdir, makedirs
import numpy as np

# Zentraler Schalter für Plot-Ausgabe
SHOW_PLOTS = True

def cut_and_analyze(file_path: str, save_dir: str, u_rated: float = 3.0):
    try:
        file_name = basename(file_path)

        # Daten laden
        data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=0.01)
        signal = data_loader.signal_data

        print(f'Signal geladen: {file_name}')

        # Holding Signal extrahieren
        holding_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated, cutaway=0.2)
        _, starttime = holding_signal.get_start_and_end_time()

        print(f'Holding Signal extrahiert.')

        # unloaing:
        unloading_signal = get_unloading_signal(signal, rated_voltage=u_rated, low_level=0.6, high_level=0.95)
        # linear fit:
        unloading_parameter = polynomial_fit(unloading_signal, order=1)
        # get time, where fitted unload signal is rated voltage
        rated_time = (u_rated - unloading_parameter[1]) / unloading_parameter[0]

        print(f'Unloading Signal mit rated_time: {rated_time:.3f}s ')

        # holding voltage berechnen
        holding_voltage = polynomial_fit(holding_signal, order=0)

        # seperiertes Signal für die Peak-Detection
        peak_detection_signal = SignalCutter(signal).cut_time_range((rated_time - 10, inf))

        # Peak Detection
        processor = PeakDetectionProcessor(peak_detection_signal, holding_signal, rated_time=rated_time, sigma_threshold=0.85)
        processor.compute_standard_deviation()
        peak_index, peak_time, peak_value, peak_mean, threshold = processor.detect_peaks()
        
        print(f'{file_name}: Peak gefunden bei {peak_time:.3f}s mit Wert {peak_value:.3f}.')

        # Unloading Signal extrahieren
        after_peak_signal = SignalCutter(signal).cut_time_range((peak_time, inf))
        unloading_signal = SignalCutter(after_peak_signal).cut_by_value("r>", 0.4 * peak_value)


        # Unloading Parameter berechnen
        unloading_parameter = polynomial_fit(unloading_signal, order=3)

        # U3 berechnen
        peak_eval_value = evaluate_polynomial(unloading_parameter, peak_time)
        u3 = peak_value - peak_eval_value

        # Name modifizieren
        name_parts = file_name.split('_')[:-1]
        name_parts = [n for n in name_parts if n != 'Testaufbau']
        name_parts.append('cut')
        name = '_'.join(name_parts)

        # Speichern
        save_path = join(save_dir, name + '.csv')
        header = {
            'holding_voltage': holding_voltage,
            'unloading_parameter': unloading_parameter,
            'peak_time': peak_time,
            'peak_value': peak_value,
            'peak_mean': peak_mean,
            'plus_minus_toleranz': threshold,
            'U3': u3
        }

        after_peak_signal.get_derivative()
        saver = SignalDataSaver(
            signal_data=after_peak_signal,
            filename=save_path,
            header_info=header
        )
        saver.save_to_csv()
        try:
            plot_results(file_name, signal, processor.peak, processor.outliers, rated_time)
        except Exception as e:
            print(f"Fehler bei Plotten von {file_path}: {e}")

    except Exception as e:
        print(f"Fehler bei Datei {file_path}: {e}")


def plot_results(file, signal, peak, outliers, rated_time):
        """Visualisiert die Daten, gefilterten Werte und markiert Peaks."""
        time_range = abs(peak['time'] - rated_time) * 2
        relevant_time_range = [rated_time , rated_time + time_range]
        relevant_indizes = []
        range_idx = 0
        for i, t in enumerate(signal.data["time"]):
            if relevant_time_range[range_idx] <= t:
                relevant_indizes.append(i)
                range_idx += 1
                if range_idx > 1:
                    break
        relevant_value_range = [signal.data["value"][relevant_indizes[0]], signal.data["value"][relevant_indizes[1]]]
        print(f'relevant_time_range: {relevant_time_range}')
        print(f'relevant_value_range: {relevant_value_range}')  

        plt.figure(figsize=(10, 5))

        plt.xlim(relevant_time_range)
        plt.ylim([min(relevant_value_range)/1.002, max(relevant_value_range)*1.002]) 
        plt.plot(signal.data["time"], signal.data["value"], label="Originaldaten", alpha=0.5)

        if outliers:
            outlier_indices, outlier_values = zip(*outliers)
            plt.scatter(signal.data["time"][list(outlier_indices)], outlier_values, color='red', label="Peaks", zorder=3)

        # Markierung des gefundenen Peaks
        plt.axvline(peak["time"], color='green', linestyle='--', label="Peak-Zeitpunkt")
        plt.axhline(peak["value"], color='green', linestyle='--', label="Peak-Wert")
        plt.axhline(peak["mean"], color='blue', linestyle='--', label="Peak-Mittelwert")
        plt.axhline(peak["mean"] + peak["threshold"], color='orange', linestyle='--', label="Threshold")
        plt.axhline(peak["mean"] - peak["threshold"], color='orange', linestyle='--')
        
        plt.legend()
        plt.grid(True)
        plt.xlabel("Zeit (s)")
        plt.ylabel("Signalwert")
        plt.title("Peak-Detection für {}".format(file))


def process_folder():
    # Tkinter Dialog initialisieren und verstecken
    root = Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    if not folder_path:
        print("Kein Ordner ausgewählt. Programm wird beendet.")
        return

    cut_data_folder = join(folder_path, "cut_data")
    if not exists(cut_data_folder):
        makedirs(cut_data_folder)

    file_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.csv')]

    for file_name in file_list:
        file_path = join(folder_path, file_name)
        cut_and_analyze(file_path, save_dir=cut_data_folder)

    if SHOW_PLOTS:
        show()


def process_single_file():
    # Tkinter Dialog initialisieren und verstecken
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        print("Keine Datei ausgewählt. Programm wird beendet.")
        return

    folder_path = dirname(file_path)
    cut_data_folder = join(folder_path, "cut_data")
    if not exists(cut_data_folder):
        makedirs(cut_data_folder)

    cut_and_analyze(file_path, save_dir=cut_data_folder)

    if SHOW_PLOTS:
        show()


if __name__ == '__main__':
    # Hier kannst du steuern, welche Funktion ausgeführt werden soll:
    # process_folder()
    process_single_file()