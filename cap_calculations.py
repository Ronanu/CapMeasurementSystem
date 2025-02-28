from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal, cut_basic_signal_nicely
from cap_signal_processing import polynomial_fit, evaluate_polynomial, interpolate_signal, holding_and_unloading
from signal_transformations import SignalDataLoader, SignalCutter, SignalData, PlotVoltageAndCurrent, SignalDataSaver
import matplotlib.pyplot as plt
from rename_files import parse_filename
import os
import numpy as np


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

    difference = np.array(interresting_voltage) - np.array(ideal_voltages) 
    # get time of max difference:
    max_diff_idx = np.argmax(np.abs(difference))
    max_diff_time = interresting_time[max_diff_idx]

    condensed_signal = SignalCutter(signal).cut_time_range((max_diff_time, unloading_end_time)) 

    # zweiter fitting Durchggang
    new_unloading_signal_fit = SignalCutter(signal).cut_time_range((max_diff_time+0.0, unloading_end_time))
    new_unloading_parameter = polynomial_fit(new_unloading_signal_fit, order=order)
    print(f'new_unloading_parameter={new_unloading_parameter}')
    new_ideal_voltages = [ideal_voltage(holding_voltage, new_unloading_parameter, t) for t in interresting_time]
    new_difference = np.array(interresting_voltage) - np.array(new_ideal_voltages)
    new_max_diff_idx = np.argmax(np.abs(new_difference))
    new_max_diff_time = interresting_time[new_max_diff_idx]
    new_condensed_signal = SignalCutter(signal).cut_time_range((new_max_diff_time, unloading_end_time))

    if not(max_diff_time == new_max_diff_time):
        raise ValueError('max_diff_time and new_max_diff_time are not equal')
    
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        axes[0].plot(signal.get_data()['time'], signal.get_data()['value'], label='Original Signal ' + name,
                      linewidth=4, alpha=0.5)
        axes[0].plot(interresting_time, interresting_voltage, label='interresting Signal')
        axes[0].plot(interresting_time, ideal_voltages, label='Ideal Voltage')
        axes[0].plot(interresting_time, new_ideal_voltages, label='New Ideal Voltage')
        axes[1].plot(interresting_time, difference, label='Difference')
        axes[1].plot(interresting_time, new_difference, label='New Difference')
        axes[2].plot(condensed_signal.get_data()['time'], condensed_signal.get_data()['value'], label='Condensed',
                     linewidth=3, alpha=0.5)
        axes[2].plot(new_condensed_signal.get_data()['time'], new_condensed_signal.get_data()['value'], label='New Condensed')
        for a in axes:
            a.legend(loc='best')
            a.grid()
        plt.legend()
    
    return new_condensed_signal, new_unloading_parameter, holding_voltage, max_diff_time
        


if __name__ == '__main__':

    folder_path = 'csv_files/'

    u_rated = 3
    esr_rated = 0.022
    i_cc = 3
    all_i_dc = {'B2': 3, 'A2': 0.6, 'A2esr': 0.06}

    all_files = os.listdir(folder_path)
            # Filtere nur Dateien mit der Endung '.picolog'
    csv_files = [f for f in all_files if f.endswith('.csv')]
    csv_files = ['MAL2_1A2esr.csv'] # ,'MAL2_2.csv','MAL2_3.csv', 'MAL2_1A2esr.csv', 'MAL2_1A2.csv']

    for file in csv_files:
        file_path = folder_path + file

        cap_nr, method, special, name = parse_filename(file, '.csv')
        print(f'Processing file {file}: nr={cap_nr}, method={method}, special={special}')
        i_dc = all_i_dc[method]
        print(f'i_dc={i_dc}')

        new_condensed_signal, new_unloading_parameter, holding_voltage, max_diff_time = get_condensed_signal(
            file=file_path,
            name = name, 
            u_rated= u_rated,
            plot=True)
        
        file_path = folder_path + 'condensed' + file
        header = {
            'holding_voltage': holding_voltage,
            'unloading_parameter': new_unloading_parameter,
            'max_diff_time': max_diff_time
        }
        new_condensed_signal.get_derivative()
        saver = SignalDataSaver(new_condensed_signal, file_path, header)
        saver.save_to_csv()
    plt.show()
