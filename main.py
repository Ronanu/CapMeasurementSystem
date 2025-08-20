from calcParamsCurrent import *
from getFileNameInfos import getFileNameInfos


if __name__ == "__main__":
    
    filename = "C_B1_DUT3_V1_Vishay_Testaufbau_25F_09-07-2025"
    manufacturer, capacitance, typ, methode, klass = getFileNameInfos(filename)
    
    print(f"Manufacturer: {manufacturer}")
    print(f"Capacitance: {capacitance}F")
    print(f"Type: {typ}")
    print(f"Methode: {methode}")
    print(f"Klass: {klass}")

    I_charge = calc_charge_current(manufacturer, capacitance)
    print(f"Charge Current: {I_charge} A")
    I_discharge = calc_discharge_current(manufacturer, capacitance, typ, methode, klass)
    print(f"Discharge Current: {I_discharge} A")