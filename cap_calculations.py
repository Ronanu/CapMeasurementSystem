from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal, cut_basic_signal_nicely
from cap_signal_processing import polynomial_fit, evaluate_polynomial, interpolate_signal, holding_and_unloading
from signal_transformations import SignalDataLoader, SignalCutter, SignalData, PlotVoltageAndCurrent, SignalDataSaver
from peak_detection import PeakDetectionProcessor

from matplotlib.pyplot import show, subplots, legend
from rename_files import parse_filename
from os.path import dirname, basename, join
from numpy import array, abs, argmax, inf
from tkinter import filedialog, Tk



if __name__ == '__main__':

    # Tkinter Dialog initialisieren und verstecken
    root = Tk()
    root.withdraw()

    # Datei auswählen
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    # Prüfen, ob eine Datei ausgewählt wurde
    if not file_path:
        print("Keine Datei ausgewählt. Programm wird beendet.")
        exit(1)

    # Verzeichnis aus dem gewählten Datei-Pfad extrahieren
    folder_path = dirname(file_path)
    file_name = basename(file_path)

    # Parameter definieren
    u_rated = 3

    # Daten laden
    data_loader = SignalDataLoader(file_path=file_path, name='Original_Signal', sampling_interval=0.01)
    signal = data_loader.signal_data

    # Holding Signal extrahieren
    holding_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated, cutaway=0.2)
    _, starttime = holding_signal.get_start_and_end_time()  # get endtime of holding signal as starttime for peak detection

    # holding voltage berechnen
    holding_voltage = polynomial_fit(holding_signal, order=0)

    peak_detection_signal = SignalCutter(signal).cut_time_range((starttime, inf))
    peak_detection_signal.plot_signal()

    # Peak Detection
    counter = 0
    peak_found = False
    processor = PeakDetectionProcessor(peak_detection_signal, holding_signal, sigma_threshold=0.55)
    while not peak_found:
        processor.high_pass_filter()
        processor.compute_standard_deviation()
        peak_index, peak_time, peak_value, peak_mean, threshold = processor.detect_peaks()
        if peak_index > 0:
            peak_found = True
            print(f'Peak gefunden bei {peak_time} mit Wert {peak_value} nach {counter} Versuchen.')
        else:
            if counter > 50:
                print(f'Abbruch nach {counter} Versuchen. Kein Peak gefunden.')
            counter += 1
            current_threshold = processor.sigma_threshold
            processor.sigma_threshold = current_threshold * 0.9
            print(f'Peak nicht gefunden. Neuer Threshold: {processor.sigma_threshold}')

    processor.plot_results()
    
    # Unloading Signal extrahieren
    after_peak_signal = SignalCutter(signal).cut_time_range((peak_time, inf))
    unloading_signal = SignalCutter(after_peak_signal).cut_by_value("r>", 0.4 * peak_value)
    unloading_signal.plot_signal()

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
    save_path = join(folder_path, name + '.csv')
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

    # Falls geplottet wurde, anzeigen
    show()
