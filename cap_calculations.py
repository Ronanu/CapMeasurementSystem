from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal, cut_basic_signal_nicely
from cap_signal_processing import polynomial_fit, evaluate_polynomial, interpolate_signal, holding_and_unloading
from signal_transformations import SignalDataLoader, SignalCutter, SignalData, PlotVoltageAndCurrent, SignalDataSaver


from matplotlib.pyplot import show, subplots, legend
from rename_files import parse_filename
from os.path import dirname, basename, join
from numpy import array, abs, argmax, inf
from tkinter import filedialog, Tk

def ideal_voltage(holding_voltage, unloading_coeff, time):
    value = evaluate_polynomial(unloading_coeff, time)
    if value > holding_voltage:
        return holding_voltage
    return value


def get_condensed_signal(file, name, u_rated, order=3, plot=False):
    
    data_loader = SignalDataLoader(file_path=file, name=name, sampling_interval=0.01)

    signal = data_loader.signal_data
    

    holding_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated)
    holding_voltage = polynomial_fit(holding_signal, order=0)[0]
    print(f'holding_voltage={holding_voltage}')
    unloading_signal = get_unloading_signal(signal, rated_voltage=u_rated, low_level=0.4, high_level=0.8)
    unloading_parameter = polynomial_fit(unloading_signal, order=order)
    print(f'unloading_parameter={unloading_parameter}')
    
    holding_end_time = holding_signal.get_start_and_end_time()[1]
    unloading_end_time = unloading_signal.get_start_and_end_time()[1]

    interresting_signal = SignalCutter(signal).cut_time_range((holding_end_time, unloading_end_time))
    interresting_time = interresting_signal.get_data()['time']
    ideal_voltages = [ideal_voltage(holding_voltage, unloading_parameter, t) for t in interresting_time]
    interresting_voltage = interresting_signal.get_data()['value']  

    difference = array(interresting_voltage) - array(ideal_voltages) 
    # get time of max difference:
    max_diff_idx = argmax(abs(difference))
    max_diff_time = interresting_time[max_diff_idx]

    condensed_signal = SignalCutter(signal).cut_time_range((max_diff_time, unloading_end_time)) 

    # zweiter fitting Durchggang
    new_unloading_signal_fit = SignalCutter(signal).cut_time_range((max_diff_time+0.0, unloading_end_time))
    new_unloading_parameter = polynomial_fit(new_unloading_signal_fit, order=order)
    print(f'new_unloading_parameter={new_unloading_parameter}')
    new_ideal_voltages = [ideal_voltage(holding_voltage, new_unloading_parameter, t) for t in interresting_time]
    new_difference = array(interresting_voltage) - array(new_ideal_voltages)

    new_max_diff_idx = argmax(abs(new_difference))
    new_max_diff_time = interresting_time[new_max_diff_idx]
    new_condensed_signal = SignalCutter(signal).cut_time_range((new_max_diff_time, inf))

    max_diff_esr = new_difference[new_max_diff_idx]

    if not(max_diff_time == new_max_diff_time):
        raise ValueError('max_diff_time and new_max_diff_time are not equal')
    
    if plot:
        _, axes = subplots(3, 1, figsize=(10, 15), sharex=True)
        axes[0].plot(signal.get_data()['time'], signal.get_data()['value'], label='Original Signal ' + name,
                      linewidth=4, alpha=0.5)
        axes[0].plot(interresting_time, interresting_voltage, label='interresting Signal')
        axes[0].plot(interresting_time, ideal_voltages, label='Ideal Voltage', linewidth = 1)
        axes[0].plot(interresting_time, new_ideal_voltages, label='New Ideal Voltage', linewidth = 1)
        axes[1].plot(interresting_time, difference, label='Difference', linewidth=3)
        axes[1].plot(interresting_time, new_difference, label='New Difference', linewidth = 1)	
        axes[2].plot(condensed_signal.get_data()['time'], condensed_signal.get_data()['value'], label='Condensed',
                     linewidth=3, alpha=0.5)
        axes[2].plot(new_condensed_signal.get_data()['time'], new_condensed_signal.get_data()['value'], label='New Condensed')
        for a in axes:
            a.legend(loc='best')
            a.grid()
        legend()
    
    return new_condensed_signal, new_unloading_parameter, holding_voltage, max_diff_time, max_diff_esr
        


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


    # Signalverarbeitung durchführen
    new_condensed_signal, new_unloading_parameter, \
    holding_voltage, max_diff_time, max_diff_esr = get_condensed_signal(
        file=file_path,
        name=file_name,
        u_rated=u_rated,
        plot=True
    )

    # Name modifizieren
    name_parts = file_name.split('_')[:-1]
    name_parts = [n for n in name_parts if n != 'Testaufbau']
    name_parts.append('cut')
    name = '_'.join(name_parts)

    # Speichern
    save_path = join(folder_path, name + '.csv')
    header = {
        'holding_voltage': holding_voltage,
        'unloading_parameter': new_unloading_parameter,
        'max_diff_time': max_diff_time,
        'U3': max_diff_esr
    }

    new_condensed_signal.get_derivative()
    saver = SignalDataSaver(
        signal_data=new_condensed_signal, 
        filename=save_path, 
        header_info=header
    )
    saver.save_to_csv()

    # Falls geplottet wurde, anzeigen
    show()
