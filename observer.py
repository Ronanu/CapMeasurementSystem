import numpy as np
import matplotlib.pyplot as plt

class KondensatorstromBeobachter:
    def __init__(self, C, Rs, dt=0.01):
        """
        Initialisiert den Beobachter.
        
        :param C: Kapazität des Kondensators (Farad)
        :param Rs: Serienwiderstand des Kondensators (Ohm)
        :param dt: Zeitdiskretisierung (Sekunden, Default = 10 ms)
        """
        self.C = C
        self.Rs = Rs
        self.dt = dt
        
        # Initialisierung der Zustände
        self.V_C_hat = 0.0  # Geschätzte Kondensatorspannung
        self.I_C_hat = 0.0  # Geschätzter Strom

        # Berechnung der Beobachterverstärkung
        self.alpha = 10 / (Rs * C)
        self.L1 = 2 * self.alpha
        self.L2 = self.alpha**2 / (self.L1 * Rs - 1 / C)
    
    def process(self, u_data):
        """
        Verarbeitet eine komplette Datenreihe von Spannungsmesswerten.
        
        :param u_data: Liste oder NumPy-Array mit Spannungswerten (V)
        :return: NumPy-Array mit geschätzten Strömen (A)
        """
        i_hat_data = np.zeros_like(u_data)
        for i, u in enumerate(u_data):
            # Beobachtergleichungen
            dV_C_hat = (self.I_C_hat / self.C + self.L1 * (u - self.V_C_hat - self.Rs * self.I_C_hat)) * self.dt
            self.V_C_hat += dV_C_hat
            self.I_C_hat = (u - self.V_C_hat) / self.Rs
            i_hat_data[i] = self.I_C_hat
        return i_hat_data
    
    def reset(self):
        """ Setzt die geschätzten Zustände zurück. """
        self.V_C_hat = 0.0
        self.I_C_hat = 0.0

    def plot_results(self, t, u_true, u_meas, i_true, i_hat):
        """
        Erstellt Subplots für Spannung und Strom.
        
        :param t: Zeitachse (Sekunden)
        :param u_true: Tatsächliche Spannung (V)
        :param u_meas: Gemessene Spannung mit Rauschen (V)
        :param i_true: Tatsächlicher Strom (A)
        :param i_hat: Geschätzter Strom (A)
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Spannung: Tatsächlich vs. Gemessen
        axs[0].plot(t, u_true, label="Tatsächliche Spannung U", color="tab:blue")
        axs[0].plot(t, u_meas, label="Gemessene Spannung U (mit Rauschen)", linestyle="dashed", color="tab:cyan")
        axs[0].set_ylabel("Spannung (V)")
        axs[0].legend()
        axs[0].grid()

        # Strom: Tatsächlich vs. Geschätzt
        axs[1].plot(t, i_true, label="Tatsächlicher Strom I", color="tab:red")
        axs[1].plot(t, i_hat, label="Geschätzter Strom I", linestyle="dashed", color="tab:orange")
        axs[1].set_xlabel("Zeit (s)")
        axs[1].set_ylabel("Strom (A)")
        axs[1].legend()
        axs[1].grid()

        plt.suptitle("Vergleich von tatsächlichen und geschätzten Werten")
        plt.show()


# Simulation
if __name__ == "__main__":
    # Systemparameter
    C = 50  # Kapazität in Farad
    Rs = 22e-3  # Innenwiderstand in Ohm
    dt = 0.01  # 10 ms Zeitschritt
    t_max = 40  # 1 Sekunde Simulation

    # Zeitachse
    t = np.arange(0, t_max, dt)

    # Tatsächliche Eingangsspannung (Sinusförmig)
    u_true = 3 * np.sin(.05 * np.pi * 0.5 * t) + 1  # Sinusförmige Spannung mit Offset

    # Spannung mit Messrauschen
    noise = np.random.normal(0, 0.01, len(t))  # Gaußsches Rauschen mit 0.5V Amplitude
    u_meas = u_true + noise

    # Tatsächlicher Strom aus der Kapazitätsgleichung
    i_true = C * np.gradient(u_true, dt)

    # Beobachter initialisieren und Daten verarbeiten
    beobachter = KondensatorstromBeobachter(C, Rs, dt)
    i_geschätzt = beobachter.process(u_meas)

    # Ergebnisse plotten
    beobachter.plot_results(t, u_true, u_meas, i_true, i_geschätzt)