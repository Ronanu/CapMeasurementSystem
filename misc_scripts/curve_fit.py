import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Definiere die Modellfunktion
def charging_curve(t, r, c):
    return r * c * (1 - np.exp(-t / (r * c)))

# Erstelle synthetische Testdaten
true_r = 0.22  # Wahre Werte für r
true_c = 50  # Wahre Werte für c
t_data = np.linspace(0, 10, 500)  # Zeitwerte
y_data = charging_curve(t_data, true_r, true_c) + np.random.normal(0, 0.5, size=t_data.shape)  # Mit Rauschen

# Fitte die Funktion an die Daten
popt, pcov = curve_fit(charging_curve, t_data, y_data, p0=[0.1, 55])  # Startwerte für r und c

# Extrahiere geschätzte Parameter
estimated_r, estimated_c = popt

# Ergebnisse ausgeben
print(f"Geschätzte Werte: r = {estimated_r:.4f}, c = {estimated_c:.4f}")

# Ergebnisse visualisieren
plt.scatter(t_data, y_data, label="Messwerte", color="red", s=10)
plt.plot(t_data, charging_curve(t_data, *popt), label=f"Fit: r={estimated_r:.2f}, c={estimated_c:.2f}", color="blue")
plt.xlabel("Zeit t")
plt.ylabel("f(t)")
plt.legend()
plt.show()