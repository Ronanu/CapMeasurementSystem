import numpy as np
import matplotlib.pyplot as plt

class RCEntladeBeobachter:
    def __init__(self, R, C, dt, L1_real, L1_imag, L2_real, L2_imag, Uc0=5, I0=10):
        """
        Beobachter für RC-Entladung mit komplex konjugierten Polstellen.
        
        Zustand: x = [uC, i]^T
        A = [[0,    -1/C],
             [1/R, -1/(R*C)]]
        y = [1, R]*x
        L = [L1, L2]^T mit getrennten Real- und Imaginärteilen
        """
        self.R = R
        self.C = C
        self.dt = dt
        
        # Systemmatrix
        self.A = np.array([[0, -1/C],
                           [1/R, -1/(R*C)]])
        
        # Messmatrix
        self.Cm = np.array([1, R]) 
        
        # Verstärkungen mit Real- und Imaginärteil
        self.L_real = np.array([L1_real, L2_real])
        self.L_imag = np.array([L1_imag, L2_imag])
        
        # Initialwerte für den Beobachter
        self.x_hat = np.array([Uc0, I0], dtype=float)
    
    def step(self, y):
        """
        Diskrete Beobachter-Dynamik mit Euler-Schritt:
          x[k+1] = x[k] + dt * ( A*x[k] + L_real*(y - Cm*x[k]) + j*L_imag*(y - Cm*x[k]) )
        """
        e = y - self.Cm @ self.x_hat
        dx = self.A @ self.x_hat + self.L_real * e  # Realanteil
        dx += np.imag(self.L_imag) * e  # Imaginärteil skaliert
        
        self.x_hat += dx * self.dt
    
    def process(self, y_series):
        """
        Verarbeitung einer ganzen Messreihe.
        """
        xlog = np.zeros((len(y_series), 2))
        for k, y in enumerate(y_series):
            self.step(y)
            xlog[k,:] = self.x_hat
        return xlog

# --- Simulation ---
if __name__=="__main__":
    # Systemparameter
    R = 0.5
    C = 50.0
    dt = 0.01  # Kleiner Zeitschritt für bessere Genauigkeit
    t_max = 50
    t = np.arange(0, t_max, dt)

    # Reale RC-Entladung (Exponentiell)
    tau = R * C  # Zeitkonstante
    uC = 5 * np.exp(-t / tau)
    i_real = 10 * np.exp(-t / tau)
    y_true = uC + R * i_real  # Messspannung = U_C + R * I

    # Rauschen in der Messung
    noise = np.random.normal(0, 0.02, len(t))
    y_meas = y_true + noise

    # --- Beobachter mit komplexen Polstellen ---
    # Verstärkungen aus der Berechnung:
    L1_real, L1_imag = 9.5, 2.8
    L2_real, L2_imag = L1_real, L1_imag

    # Beobachter mit diesen Verstärkungen
    beob = RCEntladeBeobachter(R, C, dt, L1_real, L1_imag, L2_real, L2_imag, Uc0=5, I0=10)
    xlog = beob.process(y_meas)

    # --- Plot der Ergebnisse ---
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.plot(t, y_true, 'b-', label="u_true = uC+R*i")
    plt.plot(t, y_meas, 'c--', label="y_meas (Rauschen)")
    plt.ylabel("Spannung (V)")
    plt.grid(); plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t, i_real, 'r-', label="i_true")
    plt.plot(t, xlog[:,1], 'orange', label="i_hat (Beobachter)")
    plt.xlabel("Zeit (s)"); 
    plt.ylabel("Strom (A)")
    plt.grid(); plt.legend()
    
    plt.suptitle("RC-Entladung mit Beobachter (komplex konjugierte Pole)")
    plt.show()
