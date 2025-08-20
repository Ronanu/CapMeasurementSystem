
from ui import run_app

def main():
    run_app()

if __name__ == "__main__":
    main()
from calcParamsCurrent import *
from getFileNameInfos import getFileNameInfos

# Testaufruf    
filename = "ESR_A4_DUT2_V12_Eaton_Testaufbau_25F_01-07-2025"
manufacturer, capacitance, typ, methode, klass, dut, version = getFileNameInfos(filename)
if manufacturer is None or capacitance is None or typ is None or methode is None or klass is None or dut is None or version is None:
    raise ValueError("Invalid filename format. Please ckeck the filename structure.")


print(f"Manufacturer: {manufacturer}")
print(f"Capacitance: {capacitance}F")
print(f"Type: {typ}")
print(f"Methode: {methode}")
print(f"Klass: {klass}")
print(f"DUT: {dut}")
print(f"Version: {version}")

I_charge = calc_charge_current(manufacturer, capacitance)
print(f"Charge Current: {I_charge} A")
I_discharge = calc_discharge_current(manufacturer, capacitance, typ, methode, klass)
print(f"Discharge Current: {I_discharge} A")
