import numpy as np
import matplotlib.pyplot as plt

def i_func(t):
    """
    Stromverlauf ähnlich zur Lade- und Entladekurve eines Kondensators:
    - 0 bis 1 Sekunde: Wert = 0
    - 1 bis 6 Sekunden: Exponentielles Ansteigen bis ~1
    - 6 bis 16 Sekunden: Wert bleibt ~1
    - Ab 16 Sekunden: Exponentielle Abnahme auf 0
    """
    tau = 1.0  # Zeitkonstante für exponentielles Wachstum
    tau_decay = 1.5  # Zeitkonstante für exponentielle Entladung

    if np.isscalar(t):
        if t < 1:
            return 0.0
        elif t < 6:
            return 1 - np.exp(-(t - 1) / tau)  # Exponentielles Wachstum
        elif t < 16:
            return 1.0
        else:
            return np.exp(-(t - 16) / tau_decay)  # Exponentielle Abnahme
    else:
        t = np.asarray(t)
        values = np.zeros_like(t)
        values[(t >= 1) & (t < 6)] = 1 - np.exp(-(t[(t >= 1) & (t < 6)] - 1) / tau)
        values[(t >= 6) & (t < 16)] = 1.0
        values[t >= 16] = np.exp(-(t[t >= 16] - 16) / tau_decay)
        return values
    
if __name__ == "__main__":    
    # Test mit einer grafischen Darstellung
    t_values = np.linspace(0, 20, 1000)
    i_values = i_func(t_values)

    plt.plot(t_values, i_values, label="i_func(t)")
    plt.xlabel("Zeit (s)")
    plt.ylabel("i(t)")
    plt.title("Exponentielle Lade- und Entladekurve")
    plt.grid()
    plt.legend()
    plt.show()