from calcParamsCurrent import *
from getFileNameInfos import getFileNameInfos





# Testaufruf    
filename = "ESR_B1_DUT2_V1_WuerthElektronik_Testaufbau_25F_29-07-2025"
manufacturer, capacitance, typ, methode, klass = getFileNameInfos(filename)
if manufacturer is None or capacitance is None or typ is None or methode is None or klass is None:
    raise ValueError("Invalid filename format. Please ckeck the filename structure.")


print(f"Manufacturer: {manufacturer}")
print(f"Capacitance: {capacitance}F")
print(f"Type: {typ}")
print(f"Methode: {methode}")
print(f"Klass: {klass}")

I_charge = calc_charge_current(manufacturer, capacitance)
print(f"Charge Current: {I_charge} A")
I_discharge = calc_discharge_current(manufacturer, capacitance, typ, methode, klass)
print(f"Discharge Current: {I_discharge} A")