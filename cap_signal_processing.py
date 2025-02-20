from signal_transformations import SignalDataLoader, SignalData, SignalCutter
from signal_transformations import PlotVoltageAndCurrent
from signal_transformations import MedianFilter, MovingAverageFilter, ConvolutionSmoothingFilter
import numpy as np
import matplotlib.pyplot as plt

def cut_basic_signal_nicely(signal, rated_voltage=3):
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.1 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.1 * rated_voltage)
    starttime = first_cut.get_start_and_end_time()[0]
    endtime = second_cut.get_start_and_end_time()[1]
    return SignalCutter(signal).cut_time_range((starttime, endtime))

def get_holding_voltage_signal(filename, rated_voltage, cutaway=0.5):
    signal = SignalDataLoader(filename, "Original Signal").signal_data
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.95 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.95 * rated_voltage)
    start_time, end_time = second_cut.get_start_and_end_time()
    time_diff = end_time - start_time
    cut = cutaway * time_diff 
    third_cut = SignalCutter(second_cut).cut_time_range((start_time + cut * 0.7, end_time - cut * 0.3))
    return third_cut

def get_unloading_signal(filename, rated_voltage, low_level=0.4, high_level=0.8):
    signal = SignalDataLoader(filename, "Original Signal").signal_data
    first_cut = SignalCutter(signal).cut_by_value("l>", 1.1* high_level * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", low_level * rated_voltage)
    third_cut = SignalCutter(second_cut).cut_by_value("l<", high_level * rated_voltage)
    return third_cut


def polynomial_fit(signal, order=2):
    coeff = np.polyfit(signal.get_data()["time"], signal.get_data()["value"], order)
    return coeff

def evaluate_polynomial(coeff, x):
    return np.polyval(coeff, x)

def interpolate_signal(x_values, y_values, order=2): 
    func = np.poly1d(np.polyfit(x_values, y_values, order))
    return func

def holding_and_unloading():
    folder_path = 'csv_files/'
    file_name = 'MAL2_5A2esr.csv'
    file = folder_path + file_name
    u_rated = 3
    basic_signal = SignalDataLoader(file, "Original Signal").signal_data
    holding_voltage_signal = get_holding_voltage_signal(file, rated_voltage=u_rated)
    holding_voltage_coeff = polynomial_fit(holding_voltage_signal, order=0)
    print(f'holding_voltage_coeff: {holding_voltage_coeff}')

    unloading_signal = get_unloading_signal(file, rated_voltage=u_rated)
    unloading_coeff = polynomial_fit(unloading_signal, order=2)
    print(f'unloading_coeff: {unloading_coeff}')
    
    PlotVoltageAndCurrent(
        voltage_signals=[basic_signal, holding_voltage_signal, unloading_signal],
        current_signals=[holding_voltage_signal.get_derivative_signal() * 50, unloading_signal.get_derivative_signal() * 50]
    )

if __name__ == '__main__':
    holding_and_unloading()
    plt.show()