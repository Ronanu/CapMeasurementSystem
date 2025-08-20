import sympy as sp
U_sp, C_n = sp.symbols('U_R C_n')


# Manufacturer's ESR values for different capacitances
# TODO: Add ESR values for more capacitances
ESR = {"Vishay" : {"15F" : 0.034}, 
       "Maxwell" : {"15F" : 0.025}, 
       "Sech" : {"15F" : 0.025},
       "Eaton" : {"15F" : 0.018},
       "Würth Elektronik" : {"15F" : 0.025},
       "Kyocera" : {"15F" : 0.05}
       }
# Manufacturer's rated voltage for different capacitors
U_R = {"Vishay" : 3, 
       "Maxwell" : 3, 
       "Sech" : 3,
       "Eaton" : 3,
       "Würth Elektronik" : 2.7,
       "Kyocera" : 3
       }

def get_esr(manufacturer, capacitance):
    
    """
    Get the ESR value for a given manufacturer and capacitance.
    
    :param manufacturer: Manufacturer name as a string.
    :param capacitance: Capacitance value as a string (e.g., "15F").
    :return: ESR value as a float, or None if not found.
    """
    try:
        return ESR[manufacturer][capacitance]
    except KeyError:
        return None
    
norm_A = {"C": {"1": "1*C_n",
                "2": "0.4*C_n*U_R",
                "3": "4*C_n*U_R",
                "4": "40*C_n*U_R"},
        "ESR": {"1": "10*C_n",
                "2": "4*C_n*U_R",
                "3": "40*C_n*U_R",
                "4": "400*C_n*U_R"}}



def calc_charge_current(manuf, cap):
    """
    Calculate the charge current for a capacitor.
    I_c = U_R/(38*ESR)
    """ 
    return  U_R[manuf] / (38 * get_esr(manuf, cap))   

def calc_discharge_current(manuf, cap, method, klass):
    """
    Calculate the discharge current for a capacitor.
    """
    if method == "B":
        return U_R[manuf] / (40 * get_esr(manuf, cap))
    else:
        pass  # TODO
    return 






if __name__ == "__main__":
    pass
    # Example usage
    # manufacturer = "Maxwell"
    # capacitance = "15F"
    
    # esr_value = get_esr(manufacturer, capacitance)
    # if esr_value is not None:
    #     charge_current = calc_charge_current(manufacturer, capacitance)
    #     print(f"ESR for {manufacturer} {capacitance}: {esr_value} Ohms")
    #     print(f"Charge current: {charge_current:.3f} A")
    # else:
    #     print(f"ESR value for {manufacturer} {capacitance} not found.")