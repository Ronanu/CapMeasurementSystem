import sympy as sp
U_sp, C_n = sp.symbols('U_R C_n')

# Manufacturer's ESR values for different capacitances
# TODO: Add ESR values for more capacitances
ESR = {"Vishay" : {"25" : 0.034}, 
       "Maxwell" : {"25" : 0.025}, 
       "Sech" : {"25" : 0.025},
       "Eaton" : {"25" : 0.018},
       "WuerthElektronik" : {"25" : 0.025},
       "Kyocera" : {"25" : 0.05}
       }
# Manufacturer's rated voltage for different capacitors
U_R = {"Vishay" : 3, 
       "Maxwell" : 3, 
       "Sech" : 3,
       "Eaton" : 3,
       "WuerthElektronik" : 2.7,
       "Kyocera" : 3
       }


norm_A = {"C": {"1": "1*C_n",
                "2": "0.4*C_n*U_R",
                "3": "4*C_n*U_R",
                "4": "40*C_n*U_R"},
        "ESR": {"1": "10*C_n",
                "2": "4*C_n*U_R",
                "3": "40*C_n*U_R",
                "4": "400*C_n*U_R"}}

def get_U_R(manufacturer):
    """
    Get the rated voltage for a given manufacturer.
    
    :param manufacturer: Manufacturer name as a string.
    :return: Rated voltage as a float, or None if not found.
    """
    try:
        return U_R[manufacturer]
    except KeyError:
        raise ValueError(f"Rated voltage for {manufacturer} not found.")
    
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
        raise ValueError(f"ESR value for {manufacturer} with capacitance {capacitance} not found.")
    


def calc_charge_current(manuf, cap):
    """
    Calculate the charge current for a capacitor.
    I_c = U_R/(38*ESR)
    """ 
    try:
      I_c = round(U_R[manuf] / (38 * get_esr(manuf, cap)), 3)
    except:
        raise ValueError(f"Error calculating charge current")  
    return  I_c



def calc_discharge_current(manuf, cap, typ, methode, klass):
    """
    Calculate the discharge current for a capacitor.
    """
    if methode == "B":
        try:
          I_dc = round(U_R[manuf] / (40 * get_esr(manuf, cap)), 3)
        except:
          raise ValueError(f"Error calculating discharge current for 1B method")
        return I_dc
    else:
        U_sp = get_U_R(manuf)
        C_n = cap
        try:
          expr = sp.sympify(norm_A[typ][klass])  # Parse the formula
          I_dc = round(float(expr.evalf(subs={"C_n": C_n, "U_R": U_sp})/1000), 5)
        except:
          raise ValueError(f"Error calculating discharge current for A method")
        return I_dc


    






# if __name__ == "__main__":
    
#     # manufacturer = "Würth Elektronik"
#     # capacitance = "25F"
#     # print(str(calc_discharge_current(manufacturer, 25, "C", "A", "1")) + "A")
    
#     # esr_value = get_esr(manufacturer, capacitance)
#     # if esr_value is not None:
#     #     charge_current = calc_charge_current(manufacturer, capacitance)
#     #     print(f"ESR for {manufacturer} {capacitance}: {esr_value} Ohms")
#     #     print(f"Charge current: {charge_current:.3f} A")
#     # else:
#     #     print(f"ESR value for {manufacturer} {capacitance} not found.")
#     pass