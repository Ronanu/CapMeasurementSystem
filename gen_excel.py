import pandas as pd

# Beispiel-Daten
data = {
    "Parameter": ["Spannung", "Stromstärke", "Kapazität", "Temperatur"],
    "Wert": [3.7, 2.0, 10.0, 25.0],  # Diese Spalte ist editierbar
    "Einheit": ["V", "A", "F", "°C"]
}

df = pd.DataFrame(data)

# Speichern als Excel
df.to_excel("Messparameter.xlsx", index=False, engine="openpyxl")
print("Beispiel-Excel-Datei 'Messparameter.xlsx' wurde erstellt.")
