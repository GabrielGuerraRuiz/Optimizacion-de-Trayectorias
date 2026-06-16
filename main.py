import os
import numpy as np
from pathlib import Path
from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.track_plotter import plot_track
from src.vehicle_model import evaluar_tiempo_vuelta
from src.evaluator import generar_trayectoria
from src.evaluator import evaluar_fitness

def main():
    # 1. Seleccionar los cinco circuitos
    tracks = ["MexicoCity", "Montreal", "Monza", "Oschersleben", "Sakhir"]
    
    # Rutas de carpetas
    data_dir = Path("data")
    out_dir = Path("resultados")
    fig_dir = out_dir / "figuras"
    
    # Crear carpeta de figuras si no existe
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Iniciando selección y procesamiento de pistas...\n")

    for track_name in tracks:
        file_path = data_dir / f"{track_name}.csv"
        print(f"Procesando circuito: {track_name}...")
        
        # Cargar línea central y límites laterales
        df_raw = load_track_data(file_path)
        if df_raw is None:
            print(f" -> No se encontró {file_path}. Omitiendo.\n")
            continue
            
        # Calcular los bordes geométricos para la gráfica
        track_data = procesar_pista(df_raw)
        
        # Visualizar gráficamente la pista
        save_fig_path = fig_dir / f"{track_name}_geometria.png"
        plot_track(track_data, track_name, save_path=save_fig_path)
        print(f" -> Guardado en: {save_fig_path}\n")

        # Evaluar la línea central como prueba del modelo físico
        cx = track_data["cx"]
        cy = track_data["cy"]
        
        tiempo_base, velocidades = evaluar_tiempo_vuelta(cx, cy)
        
        # Cálculos de conversión a km/h
        vel_promedio_ms = np.mean(velocidades)
        vel_promedio_kmh = vel_promedio_ms * 3.6
        
        print(f" -> Tiempo estimado en línea central: {tiempo_base:.2f} segundos")
        print(f" -> Velocidad promedio: {vel_promedio_ms:.2f} m/s ({vel_promedio_kmh:.2f} km/h)")

        # --- PRUEBA DE INTERPOLACIÓN ---
        print("\n--- Pruebas del Motor de Interpolación ---")
        
        # 1. Simular una partícula "cero" (sin desplazamientos laterales)
        # Creamos un arreglo de ceros del mismo tamaño que los puntos de la pista
        particula_cero = np.zeros(len(track_data["cx"]))
        
        # 2. Generar la trayectoria hiper-densa y suave
        x_suave, y_suave, s_denso = generar_trayectoria(
            track_data["cx"], 
            track_data["cy"], 
            track_data["nx"], # Asegúrate de que procesar_pista devuelva nx y ny
            track_data["ny"], 
            particula_cero
        )
        
        # 3. Evaluar físicamente la nueva línea continua
        tiempo_suave, v_max_suave, radios_suaves = evaluar_fitness(x_suave, y_suave)
        
        vel_suave_ms = np.mean(v_max_suave)
        vel_suave_kmh = vel_suave_ms * 3.6
        
        print(f" -> Puntos discretos originales: {len(track_data['cx'])}")
        print(f" -> Puntos interpolados continuos: {len(x_suave)}")
        print(f" -> Tiempo estimado (Trayectoria Suave): {tiempo_suave:.2f} segundos")
        print(f" -> Velocidad promedio (Suave): {vel_suave_ms:.2f} m/s ({vel_suave_kmh:.2f} km/h)")

    print("Se han procesado las pistas seleccionadas")

if __name__ == "__main__":
    main()