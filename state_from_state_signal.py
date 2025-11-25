from signal_transformations import SignalDataLoader, SignalData, SignalCutter
from signal_transformations import PlotVoltageAndCurrent
from numpy import array, abs, argmax, inf, polyfit, polyval, poly1d
from matplotlib.pyplot import show, subplots, legend
from log import logger
from io_ops import load_signal
# wenn logger noch nicht importiert wurde, dann importieren


def get_times_from_state_signal(state_signal: SignalData):
    threscold_low = 0.01
    threscold_high = 0.995 * max(state_signal.get_data()["value"])

    first_cut = SignalCutter(state_signal).cut_by_value("l>", threscold_low)
    start_time, _ = first_cut.get_start_and_end_time()
    second_cut = SignalCutter(first_cut).cut_by_value("r>", threscold_high)
    _, end_time = second_cut.get_start_and_end_time()
    third_cut = SignalCutter(second_cut).cut_by_value('l<', threscold_low)
    # third_cut.plot_signal()
    start_holding, _ = third_cut.get_start_and_end_time()
    fourth_cut = SignalCutter(third_cut).cut_by_value('l>', threscold_low)
    # fourth_cut.plot_signal()
    peak_time, _ = fourth_cut.get_start_and_end_time()
    return start_time, start_holding, peak_time, end_time

if __name__ == "__main__":
    from os.path import join, dirname
    
    base_dir = dirname(__file__)  # Ordner, in dem das Skript liegt
    file = join(base_dir, "samples", "C_A2_DUT1_V1_Vishay_Testaufbau_50F_10-11-2025.csv")
    signal, state_signal = load_signal(file)
    signal.plot_signal()
    times = get_times_from_state_signal(state_signal)
    print(times)
    show()