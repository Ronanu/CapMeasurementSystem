from cap_signal_processing import get_holding_voltage_signal, get_unloading_signal, cut_basic_signal_nicely
from cap_signal_processing import polynomial_fit, evaluate_polynomial, interpolate_signal, holding_and_unloading
from signal_transformations import SignalDataLoader, SignalCutter, SignalData, PlotVoltageAndCurrent
import matplotlib.pyplot as plt
from rename_files import parse_filename
import os
import numpy as np


def ideal_voltage(holding_voltage, unloading_coeff, time):
    value = evaluate_polynomial(unloading_coeff, time)
    if value > holding_voltage:
        return holding_voltage
    return value


folder_path = 'csv_files/'

u_rated = 3
esr_rated = 0.022
i_cc = 3
all_i_dc = {'B2': 3, 'A2': 0.6, 'A2esr': 0.06}

all_files = os.listdir(folder_path)
        # Filtere nur Dateien mit der Endung '.picolog'
csv_files = [f for f in all_files if f.endswith('.csv')]
csv_files = ['MAL2_1.csv','MAL2_2.csv','MAL2_3.csv', 'MAL2_1A2esr.csv', 'MAL2_1A2.csv']

# 3 Sunplots untereinander
fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
# im ersten plot die Spannung und die gefitte´ten kurven
# im zweiten plot die Differenz zwischen der Spannung und der gefitteten Kurve

# Definiere die Modellfunktion
def charging_curve(t, r, c, v, v0):
    return 1/(r * c) * (v-v0) * np.exp(-t / (r * c))

for file in csv_files:
    cap_nr, method, special, newName = parse_filename(file, '.csv')
    print(f'Processing file {file}: nr={cap_nr}, method={method}, special={special}')
    data_loader = SignalDataLoader(file_path=folder_path + file, name=newName, sampling_interval=0.01)
    signal = data_loader.signal_data
    i_dc = all_i_dc[method]
    print(f'i_dc={i_dc}')
    holding_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated)
    holding_voltage = polynomial_fit(holding_signal, order=0)[0]
    print(f'holding_voltage={holding_voltage}')
    unloading_signal = get_unloading_signal(signal, rated_voltage=u_rated, low_level=0.4, high_level=0.8)
    unloading_parameter = polynomial_fit(unloading_signal, order=2)
    print(f'unloading_parameter={unloading_parameter}')
    
    holding_end_time = holding_signal.get_start_and_end_time()[1]
    unloading_end_time = unloading_signal.get_start_and_end_time()[1]

    interresting_signal = SignalCutter(signal).cut_time_range((holding_end_time, unloading_end_time))
    interresting_time = interresting_signal.get_data()['time']
    ideal_voltages = [ideal_voltage(holding_voltage, unloading_parameter, t) for t in interresting_time]
    ideal_voltage_signal = SignalData(interresting_time, ideal_voltages, 'Ideal Voltage')

    interresting_voltage = interresting_signal.get_data()['value']  

    difference = np.array(interresting_voltage) - np.array(ideal_voltages) 
    # get time of max difference:
    max_diff_idx = np.argmax(np.abs(difference))
    max_diff_time = interresting_time[max_diff_idx]

    resistive_signal = SignalCutter(signal).cut_time_range((max_diff_time, max_diff_time+1)) 
    first_derivative = SignalData('first_derivative', resistive_signal.get_data()['time'], resistive_signal.get_derivative())	
    second_derivative = SignalData('Second Derivative', first_derivative.get_data()['time'], first_derivative.get_derivative())

    if True:
        new_unloading_signal_fit = SignalCutter(signal).cut_time_range((max_diff_time+0.0, unloading_end_time))
        new_unloading_parameter = polynomial_fit(new_unloading_signal_fit, order=5)
        print(f'new_unloading_parameter={new_unloading_parameter}')
        new_ideal_voltages = [ideal_voltage(holding_voltage, new_unloading_parameter, t) for t in interresting_time]
        new_ideal_voltage_signal = SignalData(interresting_time, new_ideal_voltages, 'New Ideal Voltage')
        new_difference = np.array(interresting_voltage) - np.array(new_ideal_voltages)
        new_max_diff_idx = np.argmax(np.abs(new_difference))
        new_max_diff_time = interresting_time[new_max_diff_idx]
        new_resistive_signal = SignalCutter(signal).cut_time_range((new_max_diff_time, new_max_diff_time+1))
        new_first_derivative = SignalData('first_derivative', new_resistive_signal.get_data()['time'], new_resistive_signal.get_derivative())
        new_second_derivative = SignalData('Second Derivative', new_first_derivative.get_data()['time'], new_first_derivative.get_derivative())

    axes[0].plot(interresting_time, interresting_voltage, label=newName)
    axes[0].plot(interresting_time, ideal_voltages, label='Ideal Voltage')
    axes[0].plot(interresting_time, new_ideal_voltages, label='New Ideal Voltage')
    axes[1].plot(interresting_time, difference, label=newName)
    axes[1].plot(interresting_time, new_difference, label='New Difference')
    axes[2].plot(resistive_signal.get_data()['time'], resistive_signal.get_data()['value'], label=newName)
    axes[2].plot(new_resistive_signal.get_data()['time'], new_resistive_signal.get_data()['value'], label='New '+newName)
    axes[3].plot(first_derivative.get_data()['time'], first_derivative.get_data()['value'], label=newName)
    axes[3].plot(new_first_derivative.get_data()['time'], new_first_derivative.get_data()['value'], label='New '+newName)
    axes[4].plot(second_derivative.get_data()['time'], second_derivative.get_data()['value'], label=newName)
    axes[4].plot(new_second_derivative.get_data()['time'], new_second_derivative.get_data()['value'], label='New '+newName) 
    for a in axes:
        a.legend(loc='best')
        a.grid()

plt.legend()
plt.show()
