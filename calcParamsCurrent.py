import sympy as sp
U_sp, C_n = sp.symbols('U_R C_n')

# Manufacturer's ESR values for different capacitances
# TODO: Add ESR values for more capacitances
ESR = {"vishay" : {"25" : 0.034, "50": 0.022}, 
       "maxwell" : {"25" : 0.025, "50": 0.01}, 
       "sech" : {"25" : 0.025, "50": 0.015},
       "eaton" : {"25" : 0.018},
       "wuerthelektronik" : {"25" : 0.025, "50": 0.02},
       "kyocera" : {"25" : 0.05, "50": 0.02}
       }
# Manufacturer's rated voltage for different capacitors
U_R = {"vishay" : 3, 
       "maxwell" : 3, 
       "sech" : 3,
       "eaton" : 3,
       "wuerthElektronik" : 2.7,
       "kyocera" : 3
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


    
