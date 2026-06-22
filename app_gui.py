import sys
import threading
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.evaluator import evaluar_fitness, generar_trayectoria
from src.pso import EnjambrePSO
from src.track_plotter import plot_telemetry_panel, animar_trayectoria_con_zoom

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update()

    def flush(self):
        pass

class PSODashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PSO Telemetry Dashboard - Optimización de Trayectorias")
        self.geometry("900x600")
        self.state('zoomed')
        self.configure(padx=10, pady=10)
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # Variables para guardar estado y alimentar la animación
        self.last_track_data = None
        self.last_best_traj = None
        self.last_history = None
        self.last_baseline = None
        self.last_optimized = None

        control_frame = ttk.LabelFrame(self, text="Configuración del Enjambre")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        ttk.Label(control_frame, text="Circuito a optimizar:").pack(pady=5, padx=10, anchor="w")
        self.track_var = tk.StringVar(value="MexicoCity")
        tracks = ["MexicoCity", "Montreal", "Monza", "Oschersleben", "Sakhir"]
        self.track_combo = ttk.Combobox(control_frame, textvariable=self.track_var, values=tracks, state="readonly")
        self.track_combo.pack(pady=5, padx=10, fill="x")

        ttk.Label(control_frame, text="Número de Partículas:").pack(pady=5, padx=10, anchor="w")
        self.particles_var = tk.IntVar(value=30)
        ttk.Entry(control_frame, textvariable=self.particles_var).pack(pady=5, padx=10, fill="x")

        ttk.Label(control_frame, text="Generaciones (Iteraciones):").pack(pady=5, padx=10, anchor="w")
        self.iters_var = tk.IntVar(value=30)
        ttk.Entry(control_frame, textvariable=self.iters_var).pack(pady=5, padx=10, fill="x")

        self.run_btn = ttk.Button(control_frame, text="▶ Iniciar Aprendizaje PSO", command=self.start_optimization)
        self.run_btn.pack(pady=10, padx=10, fill="x")

        self.anim_btn = ttk.Button(control_frame, text="🎬 Ver Animación Dinámica", command=self.play_animation, state=tk.DISABLED)
        self.anim_btn.pack(pady=5, padx=10, fill="x")

        ttk.Label(control_frame, text="Log de Operaciones:").pack(pady=5, padx=10, anchor="w")
        self.console = tk.Text(control_frame, height=8, wrap=tk.WORD, state=tk.NORMAL, bg="#f4f4f4")
        self.console.pack(pady=5, padx=10, fill="both", expand=True)
        
        sys.stdout = PrintLogger(self.console)

        self.result_frame = ttk.LabelFrame(self, text="Panel Dinámico de Telemetría")
        self.result_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.image_label = ttk.Label(self.result_frame)
        self.image_label.pack(fill="both", expand=True, padx=5, pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_optimization(self):
        self.run_btn.config(state=tk.DISABLED)
        self.anim_btn.config(state=tk.DISABLED)
        self.console.delete(1.0, tk.END)
        threading.Thread(target=self.run_optimization, daemon=True).start()

    def run_optimization(self):
        track_name = self.track_var.get()
        data_dir = Path("data")
        
        out_dir = Path("resultados")
        fig_dir = out_dir / "Figuras"
        traj_dir = out_dir / "Trayectorias Optimizadas"
        panel_dir = out_dir / "Panel de Telemetria"
        
        for d in [fig_dir, traj_dir, panel_dir]:
            d.mkdir(parents=True, exist_ok=True)

        config_pso = {
            'num_particulas': self.particles_var.get(),
            'num_iteraciones': self.iters_var.get(),
            'inercia_w': 0.7,
            'cognitivo_c1': 1.5,
            'social_c2': 1.5
        }

        print(f"=== Procesando {track_name} ===")
        file_path = data_dir / f"{track_name}.csv"
        
        df_raw = load_track_data(file_path)
        track_data = procesar_pista(df_raw)

        tiempo_base, v_base, _ = evaluar_fitness(track_data["cx"], track_data["cy"])
        print(f"Línea Central evaluada. Tiempo Base: {tiempo_base:.3f} s")
        print("Ejecutando Enjambre PSO...")

        optimizador = EnjambrePSO(track_data, config_pso)
        mejor_pos, mejor_tiempo, historial = optimizador.optimizar(verbose=True)

        x_opt, y_opt, _ = generar_trayectoria(
            track_data['cx'], track_data['cy'],
            track_data['nx'], track_data['ny'], mejor_pos
        )
        _, v_max_opt, _ = evaluar_fitness(x_opt, y_opt)

        print(f"\n¡Optimización Finalizada!\nMejora neta: -{tiempo_base - mejor_tiempo:.3f} s")

        # ---> GUARDADO DE VARIABLES PARA LA ANIMACIÓN <---
        self.last_track_data = track_data
        self.last_best_traj = {"x": x_opt, "y": y_opt, "v_max_ms": v_max_opt}
        self.last_history = historial
        self.last_baseline = (tiempo_base, float(np.mean(v_base)) * 3.6, v_base)
        self.last_optimized = (mejor_tiempo, float(np.mean(v_max_opt)) * 3.6, v_max_opt)

        save_path = panel_dir / f"{track_name}_gui_dashboard.png"
        plot_telemetry_panel(
            track_data=self.last_track_data,
            track_name=track_name,
            best_traj=self.last_best_traj,
            convergence_history=self.last_history,
            baseline_stats=self.last_baseline,
            optimized_stats=self.last_optimized,
            save_path=save_path
        )
        
        self.show_image(save_path)
        self.run_btn.config(state=tk.NORMAL)
        self.anim_btn.config(state=tk.NORMAL)

    def play_animation(self):
        if self.last_track_data is not None and self.last_best_traj is not None:
            print(">>> Generando Replay Dinámico...")
            animar_trayectoria_con_zoom(
                self.last_track_data, 
                self.last_best_traj,
                self.last_history,
                self.last_baseline,
                self.last_optimized
            )

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((700, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def on_closing(self):
        sys.stdout = sys.__stdout__
        self.destroy()

if __name__ == '__main__':
    app = PSODashboardApp()
    app.mainloop()