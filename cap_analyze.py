import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Datei einlesen und erste Spalte ignorieren
file_path = "MAL2_6A2.csv"
df = pd.read_csv(file_path, delimiter=",", decimal=",", quotechar='"', skipinitialspace=True, usecols=[1], names=["Spannung (V)"], skiprows=1)

# Konvertiere die Spannungswerte in numerische Werte
df["Spannung (V)"] = df["Spannung (V)"].astype(str).str.replace(",", ".").astype(float)

# Füge eine Zeitspalte hinzu (10 ms Abtastzeit)
df["Zeit (s)"] = np.arange(len(df)) * 0.01

# Beschneidung des Signals: Entferne Werte unter 500 mV
threshold_voltage = 0.5
valid_indices = df[df["Spannung (V)"] >= threshold_voltage].index

if not valid_indices.empty:
    df_trimmed = df.loc[valid_indices.min():valid_indices.max()].reset_index(drop=True)
else:
    df_trimmed = df.copy()

# Erste Ableitung berechnen (dU/dt)
df_trimmed["dU/dt"] = np.gradient(df_trimmed["Spannung (V)"], 0.01)

# Segmentierung in 1-Sekunden-Abschnitte (100 Werte pro Sekunde)
samples_per_second = 100
num_full_seconds = len(df_trimmed) // samples_per_second
mean_derivatives = []

for i in range(num_full_seconds):
    start_idx = i * samples_per_second
    end_idx = start_idx + samples_per_second
    segment = df_trimmed.iloc[start_idx:end_idx]
    mean_derivatives.append(segment["dU/dt"].mean())

# Wiederholung der Mittelwerte für alle Punkte
diff = len(df_trimmed) - len(mean_derivatives) * samples_per_second
if diff > 0:
    repeated_values = np.append(np.repeat(mean_derivatives, samples_per_second), [mean_derivatives[-1]] * diff)
elif diff < 0:
    repeated_values = np.repeat(mean_derivatives, samples_per_second)[:len(df_trimmed)]
else:
    repeated_values = np.repeat(mean_derivatives, samples_per_second)

# Sicherstellen, dass die Länge übereinstimmt
assert len(repeated_values) == len(df_trimmed), "Fehlanpassung der Längen"
df_trimmed["Average dU/dt"] = repeated_values

# Erstellung des Subplots mit Originalsignal und Ableitung
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Oberes Diagramm: Originalsignal
axs[0].plot(df_trimmed["Zeit (s)"], df_trimmed["Spannung (V)"], color="blue", label="Spannung (V)")
axs[0].set_ylabel("Spannung (V)")
axs[0].set_title("Originalsignal (beschnitten)")
axs[0].legend()
axs[0].grid()

# Unteres Diagramm: Durchschnittliche Ableitung pro Segment
axs[1].plot(df_trimmed["Zeit (s)"], df_trimmed["Average dU/dt"], color="red", label="Segmentierte mittlere Ableitung")
axs[1].set_xlabel("Zeit (s)")
axs[1].set_ylabel("Mittlere Ableitung (dU/dt)")
axs[1].set_title("Mittlere Ableitung pro Sekunde über die Zeit")
axs[1].axhline(0, color="black", linestyle="dashed", linewidth=1)
axs[1].legend()
axs[1].grid()

# Anzeige des Plots
plt.tight_layout()
plt.show()