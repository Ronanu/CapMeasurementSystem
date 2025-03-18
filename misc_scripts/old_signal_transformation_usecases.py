from signal_transformations import SignalDataLoader, SignalData, SignalCutter
from signal_transformations import PlotVoltageAndCurrent
from signal_transformations import MedianFilter, MovingAverageFilter, ConvolutionSmoothingFilter
import numpy as np
import matplotlib.pyplot as plt



def compare_raw_to_pressed():
    file_path = "MAL2_5A2esr.csv"
    raw_signal = SignalDataLoader(file_path, "Original Signal").signal_data
    file_path = "p1A2esr.csv"
    pressed_signal = SignalDataLoader(file_path, "Pressed Signal").signal_data

    current_signal = raw_signal.get_derivative_signal() * 50
    smoothed_signal = ConvolutionSmoothingFilter(raw_signal, kernel_size=23).signal_data
    smoothed_current = smoothed_signal.get_derivative_signal() * 50

    pressed_current = pressed_signal.get_derivative_signal() * 50
    pressed_smoothed = ConvolutionSmoothingFilter(pressed_signal, kernel_size=23).signal_data
    pressed_smoothed_current = pressed_smoothed.get_derivative_signal() * 50
    
    PlotVoltageAndCurrent(
        voltage_signals=[raw_signal, smoothed_signal, pressed_signal, pressed_smoothed],
        current_signals=[current_signal, smoothed_current, pressed_current, pressed_smoothed_current]
    )

def get_holding_voltage(filename, rated_voltage, do_plot=True):
    signal = SignalDataLoader(filename, "Original Signal").signal_data
    
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.95 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.95 * rated_voltage)
    start_time, end_time = second_cut.get_start_and_end_time()
    third_cut = SignalCutter(second_cut).cut_time_range((start_time + 60, end_time - 60))
    signal_data = third_cut.get_data()

    coeff = np.polyfit(signal_data["time"], signal_data["value"], 0)
    holding_voltage = coeff[0]

    print(f'holding_voltage: {holding_voltage} V')
    if do_plot:
        PlotVoltageAndCurrent(
            voltage_signals=[signal, first_cut, second_cut, third_cut],
            current_signals=[signal.get_derivative_signal() * 50, first_cut.get_derivative_signal() * 50, second_cut.get_derivative_signal() * 50]
        )
        return holding_voltage
    
def get_unloading(filename, rated_voltage, discharge_current=0.6, do_plot=True):
    signal = SignalDataLoader(filename, "Original Signal").signal_data
    first_cut = SignalCutter(signal).cut_by_value("l>", 0.95 * rated_voltage)
    second_cut = SignalCutter(first_cut).cut_by_value("r>", 0.4 * rated_voltage)
    third_cut = SignalCutter(second_cut).cut_by_value("l<", 0.8 * rated_voltage)

    order = 2
    coeff = np.polyfit(third_cut.get_data()["time"], third_cut.get_data()["value"], order)
    print(f'coeff: {coeff}')
    
    # create signal from coefficients of polynomial
    time =  third_cut.get_data()["time"]
    a, b, c = coeff
    voltage_poly_signal = SignalData("Polynomial Signal", time, np.polyval(coeff, time))
    capacity = -discharge_current / (2 * a * time + b)

    # plot capacity signal with voltage signal on the x-axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(voltage_poly_signal.get_data()["value"], capacity, label="Capacity Signal")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Capacity (F)")
    ax.set_title("Capacity vs. Voltage")
    ax.grid()


    if do_plot:
        PlotVoltageAndCurrent(
            voltage_signals=[signal, first_cut, second_cut, third_cut, voltage_poly_signal],
            current_signals=[signal.get_derivative_signal() * 50, first_cut.get_derivative_signal() * 50, second_cut.get_derivative_signal() * 50]
        )
    return coeff, capacity, voltage_poly_signal

if __name__ == "__main__":
    
    _, cp1, v1 = get_unloading("csv_files/MAL2_5A2esr.csv", 3, do_plot=False)
    _, cp2, v2 = get_unloading("csv_files/p1A2esr.csv", 3, do_plot=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(v1.get_data()["value"], cp1, label="Capacity 1 Signal")
    ax.plot(v2.get_data()["value"], cp2, label="Capacity 2 Signal")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Capacity (F)")
    ax.set_title("Capacity vs. Voltage")
    ax.legend()
    ax.grid()

    plt.show()