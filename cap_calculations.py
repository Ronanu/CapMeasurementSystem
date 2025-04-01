from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal
from cap_signal_processing import polynomial_fit, evaluate_polynomial
from signal_transformations import SignalDataLoader, SignalCutter, SignalDataSaver
from peak_detection import PeakDetectionProcessor
from log import logger

from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from os.path import dirname, basename, join, isfile, exists
from numpy import inf
from tkinter import filedialog, Tk
from os import listdir, makedirs
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

std_def_factor = 3.0

SHOW_PLOTS = True
PLOT_PAGES = []
PDF_OUTPUT_PATH = "alle_peaks.pdf"

def cut_and_analyze_peak(file_path: str, save_dir: str, u_rated: float = 3.0):
    try:
        file_name = basename(file_path)

        # Daten laden
        data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=0.01)
        signal = data_loader.signal_data

        logger.info(f'Signal geladen: {file_name}')

        # Holding Signal extrahieren
        holding_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated, cutaway=0.2)
        _, starttime = holding_signal.get_start_and_end_time()
        # holding voltage berechnen
        holding_voltage = polynomial_fit(holding_signal, order=0)[0]

        logger.info(f'Holding Signal extrahiert. Holding Voltage: {holding_voltage:.3f}V')

        # unloaing:
        unloading_signal = get_unloading_signal(signal, rated_voltage=u_rated, low_level=0.6, high_level=0.90)
        # linear fit:
        unloading_parameter = polynomial_fit(unloading_signal, order=1)
        # get time, where fitted unload signal is rated voltage
        rated_time = (u_rated - unloading_parameter[1]) / unloading_parameter[0]

        logger.info(f'Unloading Signal mit rated_time: {rated_time:.3f}s ')

        window_time = 10

    
        # seperiertes Signal für die Peak-Detection
        peak_detection_signal = SignalCutter(signal).cut_time_range((rated_time - window_time, rated_time))
        std_dev = np.std(peak_detection_signal.data["value"])	
        peak_linear_function = polynomial_fit(peak_detection_signal, order=1)

        signal_to_cut = SignalCutter(signal).cut_time_range((rated_time - window_time, inf))
        signal_to_cut.get_derivative()
        # nun soll das signal_to_cut von hinten nach vorne durchsucht werden, um den peak zu finden

        limit_reached = False
        threshold = std_def_factor * std_dev
        outliers = []

        for t, val , dval in zip(reversed(signal_to_cut.data["time"]), reversed(signal_to_cut.data["value"]), reversed(signal_to_cut.data["derivative"])):
            limit_value = evaluate_polynomial(peak_linear_function, t)
            if not limit_reached and val > limit_value - threshold:
                limit_reached = True
                outliers.append((t, val))
            if limit_reached:
                if dval < -0.04:
                    outliers.append((t, val))
                    continue
                else:
                    peak_time = t
                    peak_value = val
                    break
        
        logger.info(f'{file_name}: Peak gefunden bei {peak_time:.3f}s mit Wert {peak_value:.3f}.')

        # Unloading Signal extrahieren
        after_peak_signal = SignalCutter(signal).cut_time_range((peak_time, inf))
        unloading_signal = SignalCutter(after_peak_signal).cut_by_value("r>", 0.4 * peak_value)

        # mean peak_value berechnen
        mean_signal = SignalCutter(signal).cut_time_range((peak_time - 10, peak_time - 1))
        peak_mean = np.mean(mean_signal.data["value"])

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
        if SHOW_PLOTS:
            try:
                peak = {"time": peak_time, "value": peak_value, "mean": peak_mean, "threshold": threshold}
                plot_results(file_name, signal, peak, outliers, rated_time)
            except Exception as e:
                logger.warning(f"Fehler bei Plotten von {file_path}: {e}")

    except Exception as e:
        logger.warning(f"Fehler bei Datei {file_path}: {e}")


from signal_transformations import MovingAverageFilter

def plot_results(file, signal, peak, outliers, rated_time):
    """Visualisiert die Daten, gefilterten Werte und markiert Peaks, inkl. Ableitung und Glättung."""

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
    logger.debug(f'relevant_time_range: {relevant_time_range}')
    logger.debug(f'relevant_value_range: {relevant_value_range}')  

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()  # zweite y-Achse für Ableitungen

    # Achsenbereiche
    ax1.set_xlim(relevant_time_range)
    ax1.set_ylim([min(relevant_value_range)/1.002, max(relevant_value_range)*1.002]) 

    # Signal und Ableitung vorbereiten
    if signal.data["derivative"].isnull().all():
        signal.get_derivative()

    # geglättete Ableitung berechnen
    average=8
    smoothed_derivative_signal = MovingAverageFilter(signal.get_derivative_signal(), window_size=average).signal_data

    # Plots
    ax1.plot(signal.data["time"], signal.data["value"], label="Originaldaten", alpha=1, linewidth=1.5, color="tab:blue")
    ax2.plot(signal.data["time"], signal.data["derivative"], label="Ableitung", color="tab:red", alpha=0.3, linewidth=0.5)
    ax2.plot(smoothed_derivative_signal.data["time"], smoothed_derivative_signal.data["value"],
             label=f"moving average n={average}", color="tab:orange", linewidth=1.2)

    # Ausreißer und Peak markieren
    if outliers:
        outlier_times, outlier_values = zip(*outliers)
        ax1.scatter(outlier_times, outlier_values, color='red', label="Ausreißer", s=10)
        ax1.scatter(peak["time"], peak["value"], color='red', label="Peak", s=20)

    # Linien und Beschriftung
    ax1.axvline(peak["time"], color='green', linestyle='--', label="Peak-Zeitpunkt")
    ax1.axhline(peak["value"], color='green', linestyle='--', label="Peak-Wert")
    ax1.axhline(peak["mean"], color='blue', linestyle='--', label="Peak-Mittelwert")
    ax1.axhline(peak["mean"] - peak["threshold"], color='orange', linestyle='--', label="Threshold")

    # Achsen-Labels
    ax1.set_xlabel("Zeit (s)")
    ax1.set_ylabel("Signalwert (V)")
    ax2.set_ylabel("Ableitung (V/s)")

    # Nur Grid für ax1
    ax1.grid(True)
    ax2.grid(False)

    # Legenden
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f"{file} Peak-Detection")

    PLOT_PAGES.append(fig)




def process_folder():
    # Tkinter Dialog initialisieren und verstecken
    root = Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()
    if not folder_path:
        logger.warning("Kein Ordner ausgewählt. Programm wird beendet.")
        return

    cut_data_folder = join(folder_path, "cut_data")
    if not exists(cut_data_folder):
        makedirs(cut_data_folder)

    file_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.csv')]

    for file_name in file_list:
        file_path = join(folder_path, file_name)
        cut_and_analyze_peak(file_path, save_dir=cut_data_folder)

    if PLOT_PAGES:
        pdf_path = join(cut_data_folder, PDF_OUTPUT_PATH)
        with PdfPages(pdf_path) as pdf:
            for fig in PLOT_PAGES:
                pdf.savefig(fig)
                plt.close(fig)
        logger.info(f"Alle Plots gespeichert in: {pdf_path}")


def process_single_file():
    # Tkinter Dialog initialisieren und verstecken
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        logger.warning("Keine Datei ausgewählt. Programm wird beendet.")
        return

    folder_path = dirname(file_path)
    cut_data_folder = join(folder_path, "cut_data")
    if not exists(cut_data_folder):
        makedirs(cut_data_folder)

    cut_and_analyze_peak(file_path, save_dir=cut_data_folder)

    if SHOW_PLOTS:
        show()


if __name__ == '__main__':
    # Hier kannst du steuern, welche Funktion ausgeführt werden soll:
    process_folder()
    # process_single_file()