import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Supercap Testing - Signal Plot")

        # Menüleiste
        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Datei öffnen", command=self.load_file)
        file_menu.add_command(label="Beenden", command=root.quit)
        menubar.add_cascade(label="Datei", menu=file_menu)
        root.config(menu=menubar)

        # Frame für Matplotlib-Plot
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Erster leerer Plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Messsignal")
        self.ax.set_xlabel("Zeit (s)")
        self.ax.set_ylabel("Spannung (V)")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button zum Neuzeichnen des Plots
        self.plot_button = tk.Button(root, text="Neues Signal zeichnen", command=self.update_plot)
        self.plot_button.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            print(f"Geladene Datei: {file_path}")
            self.update_plot()

    def update_plot(self):
        """Simuliert das Neuzeichnen des Plots mit zufälligen Daten."""
        self.ax.clear()
        t = np.linspace(0, 10, 100)
        voltage = np.sin(t) + np.random.normal(0, 0.1, size=len(t))
        self.ax.plot(t, voltage, label="Spannung")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
