import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# -------------------
# 1) SYSTEM-KLASSE
# -------------------
class RCSystem:
    """
    Simuliert ein RC-System mit
      d/dt[vC] = i(t)/C
    Ausgang: y(t) = vC(t) + R*i(t)

    Eingabe: i(t) als frei definierbare Funktion,
    die skalare oder Array-Inputs verarbeiten kann.
    """
    def __init__(self, C, R, i_func):
        """
        :param C: Kapazität in Farad
        :param R: Serienwiderstand in Ohm
        :param i_func: Callable, i_func(t)->Strom in Ampere
                       (skalare oder arrayförmige Eingabe)
        """
        self.C = C
        self.R = R
        self.i_func = i_func
        self.t_solution = None
        self.vC_solution = None

    def _ode_system(self, t, x):
        # x[0] = vC
        # dvC/dt = i(t)/C
        return [self.i_func(t)/self.C]

    def simulate(self, t_span, x0, dt=0.001):
        """
        Löst das System-ODE von t_span[0] bis t_span[1] mit Anfangswert x0 (vC(0)).
        Speichert Zeit und Lösung in Instanzvariablen.
        """
        sol = solve_ivp(self._ode_system, t_span, [x0],
                        t_eval=np.arange(t_span[0], t_span[1]+dt, dt))
        self.t_solution = sol.t
        self.vC_solution = sol.y[0]

    def get_output(self, add_noise=False, noise_func=None):
        """
        Gibt die gemessene Gesamtspannung y(t) = vC(t) + R * i(t).
        Wenn add_noise=True und noise_func angegeben, wird Rauschen aufaddiert.
        :return: (t, y(t))
        """
        if self.t_solution is None:
            raise RuntimeError("System not yet simulated!")

        # i_func(t_solution) => shape=(len(t_solution),)
        i_vals = self.i_func(self.t_solution)
        y_clean = self.vC_solution + self.R * i_vals
        if add_noise and noise_func is not None:
            noise_vals = noise_func(self.t_solution)
            y_noisy = y_clean + noise_vals
            return self.t_solution, y_noisy
        else:
            return self.t_solution, y_clean

# -------------------
# 2) BEOBACHTER-KLASSE
# -------------------
class RCObserver:
    """
    Beobachter für das RC-System (Luenberger-ähnlich).
    Zustände: (vC_hat, i_hat)
    
    ODE:
      dvC_hat/dt = i_hat/C + L1*(y - (vC_hat + R*i_hat))
      di_hat/dt   = L2*(y - (vC_hat + R*i_hat))
    """
    def __init__(self, C, R, L1, L2):
        self.C = C
        self.R = R
        self.L1 = L1
        self.L2 = L2

        self.t_solution = None
        self.xhat_solution = None   # shape=(2, n_points)
        self.y_func = None  # wird extern gesetzt

    def _ode_observer(self, t, xhat):
        """
        xhat[0]=vC_hat, xhat[1]=i_hat
        """
        if self.y_func is None:
            raise RuntimeError("No measurement function set!")

        y = self.y_func(t)  # gemessene Spannung (Skalar)
        e = y - (xhat[0] + self.R*xhat[1])

        dvC_hat = xhat[1]/self.C + self.L1*e
        di_hat  = self.L2*e
        return [dvC_hat, di_hat]

    def set_measurement(self, t_vals, y_vals):
        """
        Erstellt aus den diskreten Daten (t_vals, y_vals)
        eine Interpolationsfunktion y_func(t).
        """
        self.y_func = interp1d(t_vals, y_vals, bounds_error=False,
                               fill_value=(y_vals[0], y_vals[-1]))

    def simulate(self, t_span, x0, dt=0.001):
        sol = solve_ivp(self._ode_observer, t_span, x0,
                        t_eval=np.arange(t_span[0], t_span[1]+dt, dt))
        self.t_solution = sol.t
        self.xhat_solution = sol.y

    def get_estimates(self):
        if self.xhat_solution is None:
            raise RuntimeError("Observer not simulated!")
        return self.t_solution, self.xhat_solution[0], self.xhat_solution[1]

# -------------------
# 3) MAIN-DEMO
# -------------------
if __name__ == "__main__":
    # Parameter
    C = 50.0
    R = 0.5
    t0, tEnd = 0.0, 50.0
    dt_sim = 0.01

    # Strom-Funktion i(t) => 2 A konstant
    # WICHTIG: Prüfen auf Skalar/Array => so vermeidest du shape-Fehler
    def i_const(t):
        if np.isscalar(t):
            return 2.0
        else:
            return np.full_like(t, 2.0)
        

    def i_funcx(t):
        """
        Gibt eine Rampe zurück, die sich wie folgt verhält:
        - 0 Sekunden bis 1 Sekunde: Wert = 0
        - 1 Sekunde bis 6 Sekunden: Linearer Anstieg von 0 auf 1
        - 6 Sekunden bis 16 Sekunden: Wert = 1
        - Ab 16 Sekunden: Wert = 0
        """
        if np.isscalar(t):
            if t < 1:
                return 0.0
            elif t < 6:
                return (t - 1) / 5  # Linearer Anstieg von 0 bis 1 in 5 Sekunden
            elif t < 16:
                return 1.0
            else:
                return 0.0
        else:
            t = np.asarray(t)  # Sicherstellen, dass t ein Array ist
            values = np.zeros_like(t)  # Standardmäßig alles auf 0 setzen
            values[(t >= 1) & (t < 6)] = (t[(t >= 1) & (t < 6)] - 1) / 5
            values[(t >= 6) & (t < 16)] = 1.0
            return values

    from stromkurven import i_func

    # 1) System
    system = RCSystem(C, R, i_func)
    vC0 = 0.0  # Startspannung
    system.simulate((t0, tEnd), x0=vC0, dt=dt_sim)

    # Rauschen
    rng = np.random.default_rng(42)
    noise_data = rng.normal(0, 0.0005, len(system.t_solution))
    def noise_func(t_array):
        # t_array shape=(n,)
        # => Return noise in same shape
        return np.interp(t_array, system.t_solution, noise_data)

    t_meas, y_meas = system.get_output(add_noise=True, noise_func=noise_func)

    # 2) Beobachter
    alpha = 10.0/(R*C)
    f = 0.0050
    L1 = f * 2*alpha
    L2 = f * alpha**2 / (L1*R - 1.0/C)

    observer = RCObserver(C, R, L1, L2)
    observer.set_measurement(t_meas, y_meas)

    xhat0 = [0.0, 0.0]  # Startwerte: vC_hat=0, i_hat=0
    observer.simulate((t0, tEnd), xhat0, dt=dt_sim)
    t_obs, vC_hat, i_hat = observer.get_estimates()

    # 3) Plotten
    # Reale Systemdaten
    t_sys = system.t_solution
    vC_sys = system.vC_solution
    i_sys = i_func(t_sys)  # shape=(len(t_sys),)
    y_clean = vC_sys + R*i_sys

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(t_sys, y_clean, 'b-', label="y_clean")
    plt.plot(t_meas, y_meas, 'c--', label="y_meas (Rauschen)")
    plt.plot(t_obs, vC_hat + R*i_hat, 'm:', label="y_hat (vC_hat+R*i_hat)")
    plt.legend()
    plt.grid()
    plt.title("Spannung")

    plt.subplot(2,1,2)
    plt.plot(t_sys, i_sys, 'r-', label="i_sys (real)")
    plt.plot(t_obs, i_hat, 'orange', label="i_hat (Beobachter)")
    plt.legend()
    plt.grid()
    plt.title("Strom")
    plt.xlabel("Zeit (s)")

    plt.tight_layout()
    plt.show()