import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.evaluator import evaluar_fitness, generar_trayectoria
from src.pso import EnjambrePSO
from src.track_plotter import plot_telemetry_panel

def main():
    tracks = ["MexicoCity", "Montreal", "Monza", "Oschersleben", "Sakhir"]
    data_dir = Path("data")
    
    # --- NUEVA ESTRUCTURA DE CARPETAS ---
    out_dir = Path("resultados")
    fig_dir = out_dir / "Figuras"
    traj_dir = out_dir / "Trayectorias Optimizadas"
    panel_dir = out_dir / "Panel de Telemetria"
    
    # Crear todas las carpetas automáticamente
    for d in [fig_dir, traj_dir, panel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    config_pso = {
        'num_particulas': 60,
        'num_iteraciones': 150,
        'inercia_w': 0.7,
        'cognitivo_c1': 1.5,
        'social_c2': 1.5
    }

    num_ejecuciones = 10

    print(f"Iniciando Evaluación Experimental ({num_ejecuciones} corridas por pista)...\n")

    for track_name in tracks:
        file_path = data_dir / f"{track_name}.csv"
        print(f"==================================================")
        print(f" PROCESANDO CIRCUITO: {track_name}")

        df_raw = load_track_data(file_path)
        if df_raw is None:
            continue
        track_data = procesar_pista(df_raw)

        tiempo_base, v_base, _ = evaluar_fitness(track_data["cx"], track_data["cy"])
        print(f" -> Tiempo Base (Línea Central): {tiempo_base:.3f} s")

        tiempos_optimos = []
        tiempos_computacionales = []
        mejor_trayectoria_global = None
        mejor_tiempo_absoluto = np.inf
        mejor_historial_convergencia = None

        for corrida in range(num_ejecuciones):
            inicio_cpu = time.time()
            optimizador = EnjambrePSO(track_data, config_pso)
            mejor_pos_corrida, mejor_tiempo_corrida, historial_convergencia = optimizador.optimizar(verbose=False)
            fin_cpu = time.time()
            tiempo_cpu = fin_cpu - inicio_cpu

            tiempos_optimos.append(mejor_tiempo_corrida)
            tiempos_computacionales.append(tiempo_cpu)

            if mejor_tiempo_corrida < mejor_tiempo_absoluto:
                mejor_tiempo_absoluto = mejor_tiempo_corrida
                mejor_trayectoria_global = mejor_pos_corrida
                mejor_historial_convergencia = historial_convergencia

            print(f"  Corrida {corrida+1}/{num_ejecuciones} | Resultado: {mejor_tiempo_corrida:.3f} s | CPU: {tiempo_cpu:.2f} s")

        t_optimos = np.array(tiempos_optimos)
        print(f"\n REPORTE ESTADÍSTICO - {track_name}:")
        print(f" -> Mejor Tiempo:    {np.min(t_optimos):.3f} s")
        print(f" -> Peor Tiempo:     {np.max(t_optimos):.3f} s")
        print(f" -> Tiempo Promedio: {np.mean(t_optimos):.3f} s")
        print(f" -> Desviación Est.: {np.std(t_optimos):.4f} s")
        print(f" -> T. Computacional Medio: {np.mean(tiempos_computacionales):.2f} s por corrida")
        print(f" -> Mejora Máxima Lograda:  -{tiempo_base - np.min(t_optimos):.3f} s")

        x_opt, y_opt, _ = generar_trayectoria(
            track_data['cx'], track_data['cy'],
            track_data['nx'], track_data['ny'],
            mejor_trayectoria_global
        )
        _, v_max_opt, _ = evaluar_fitness(x_opt, y_opt)

        df_telemetria = pd.DataFrame({
            'x': x_opt,
            'y': y_opt,
            'v_max_ms': v_max_opt
        })
        
        # Guardar en la nueva subcarpeta de trayectorias
        path_telemetria = traj_dir / f"{track_name}_trayectoria_optima.csv"
        df_telemetria.to_csv(path_telemetria, index=False)
        print(f" Trayectoria óptima exportada en: {path_telemetria}")

        # Guardar en la nueva subcarpeta de paneles
        save_panel_path = panel_dir / f"{track_name}_panel_telemetria.png"
        plot_telemetry_panel(
            track_data=track_data,
            track_name=track_name,
            best_traj={"x": x_opt, "y": y_opt, "v_max_ms": v_max_opt},
            convergence_history=mejor_historial_convergencia,
            baseline_stats=(tiempo_base, float(np.mean(v_base)) * 3.6, v_base),
            optimized_stats=(mejor_tiempo_absoluto, float(np.mean(v_max_opt)) * 3.6, v_max_opt),
            save_path=save_panel_path
        )
        print(f" Panel de telemetría guardado en: {save_panel_path}\n")

if __name__ == "__main__":
    main()