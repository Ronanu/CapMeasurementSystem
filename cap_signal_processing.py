from signal_transformations import SignalDataLoader, SignalData, SignalCutter
from signal_transformations import PlotVoltageAndCurrent
from signal_transformations import MedianFilter, MovingAverageFilter, ConvolutionSmoothingFilter
from numpy import array, abs, argmax, inf
from numpy import polyfit, polyval, poly1d
from matplotlib.pyplot import show
from matplotlib.pyplot import show, subplots, legend

def ideal_voltage(holding_voltage, unloading_coeff, time):
    value = evaluate_polynomial(unloading_coeff, time)
    if value > holding_voltage:
        return holding_voltage
    return value

def cut_basic_signal_nicely(signal, rated_voltage=3):
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.03 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.03 * rated_voltage)
    starttime = first_cut.get_start_and_end_time()[0]
    endtime = second_cut.get_start_and_end_time()[1]
    return SignalCutter(signal).cut_time_range((starttime-10, endtime+20))

def get_holding_voltage_signal(signal, rated_voltage, cutaway=0.5):
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.95 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.95 * rated_voltage)
    start_time, end_time = second_cut.get_start_and_end_time()
    time_diff = end_time - start_time
    cut = cutaway * time_diff 
    third_cut = SignalCutter(second_cut).cut_time_range((start_time + cut * 0.7, end_time - cut * 0.3))
    return third_cut

def get_unloading_signal(signal: SignalData, rated_voltage, low_level=0.4, high_level=0.8):
    first_cut = SignalCutter(signal).cut_by_value("l>", 1.1* high_level * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", low_level * rated_voltage)
    third_cut = SignalCutter(second_cut).cut_by_value("l<", high_level * rated_voltage)
    return third_cut


def polynomial_fit(signal, order=2):
    coeff = polyfit(signal.get_data()["time"], signal.get_data()["value"], order)
    return coeff

def evaluate_polynomial(coeff, x):
    return polyval(coeff, x)

def interpolate_signal(x_values, y_values, order=2): 
    func = poly1d(polyfit(x_values, y_values, order))
    return func

def holding_and_unloading(signal: SignalData, u_rated: float, do_plot=True):
    holding_voltage_signal = get_holding_voltage_signal(signal, rated_voltage=u_rated)
    holding_voltage_coeff = polynomial_fit(holding_voltage_signal, order=0)
    print(f'holding_voltage_coeff: {holding_voltage_coeff}')

    unloading_signal = get_unloading_signal(signal, rated_voltage=u_rated)
    unloading_coeff = polynomial_fit(unloading_signal, order=2)
    print(f'unloading_coeff: {unloading_coeff}')
    
    if do_plot:
        PlotVoltageAndCurrent(
            voltage_signals=[signal, holding_voltage_signal, unloading_signal],
            current_signals=[holding_voltage_signal.get_derivative_signal() * 50, unloading_signal.get_derivative_signal() * 50]
        )

def difference_method(file, name, u_rated, order=3, plot=False):
    
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
        pass
    
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
    folder_path = 'csv_files/'
    file_name = 'MAL2_5A2esr.csv'
    file = folder_path + file_name
    signal = SignalDataLoader(file, "Original Signal").signal_data
    u_rated = 3
    holding_and_unloading(signal=signal, u_rated=u_rated, do_plot=True)
    show()