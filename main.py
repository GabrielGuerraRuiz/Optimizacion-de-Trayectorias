import os
import time
import numpy as np
from pathlib import Path
from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.evaluator import evaluar_fitness, generar_trayectoria
from src.pso import EnjambrePSO

def main():
    # 1. Configuración de Pistas y Rutas
    tracks = ["MexicoCity", "Montreal", "Monza", "Oschersleben", "Sakhir"]
    data_dir = Path("data")
    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)
    
    # 2. Hiperparámetros del PSO
    config_pso = {
        'num_particulas': 30,   
        'num_iteraciones': 30,  
        'inercia_w': 0.7,       
        'cognitivo_c1': 1.5,    
        'social_c2': 1.5        
    }
    
    num_ejecuciones = 10 # REQUERIMIENTO 5: 10 ejecuciones independientes

    print(f"Iniciando Evaluación Experimental ({num_ejecuciones} corridas por pista)...\n")

    for track_name in tracks:
        file_path = data_dir / f"{track_name}.csv"
        print(f"==================================================")
        print(f" PROCESANDO CIRCUITO: {track_name}")
        
        # Carga y proceso geométrico
        df_raw = load_track_data(file_path)
        if df_raw is None:
            continue
        track_data = procesar_pista(df_raw)
        
        # Evaluación Baseline (Línea Central)
        tiempo_base, _, _ = evaluar_fitness(track_data["cx"], track_data["cy"])
        print(f" -> Tiempo Base (Línea Central): {tiempo_base:.3f} s")
        
        # Variables para estadística
        tiempos_optimos = []
        tiempos_computacionales = []
        mejor_trayectoria_global = None
        mejor_tiempo_absoluto = np.inf

        # EJECUCIÓN MÚLTIPLE (Paso 5 del Proyecto)
        for corrida in range(num_ejecuciones):
            inicio_cpu = time.time()
            
            # Instanciar y ejecutar el optimizador
            optimizador = EnjambrePSO(track_data, config_pso)
            mejor_pos_corrida, mejor_tiempo_corrida, _ = optimizador.optimizar(verbose=False)
            
            fin_cpu = time.time()
            tiempo_cpu = fin_cpu - inicio_cpu
            
            tiempos_optimos.append(mejor_tiempo_corrida)
            tiempos_computacionales.append(tiempo_cpu)
            
            # Guardar la mejor trayectoria absoluta de las 10 corridas
            if mejor_tiempo_corrida < mejor_tiempo_absoluto:
                mejor_tiempo_absoluto = mejor_tiempo_corrida
                mejor_trayectoria_global = mejor_pos_corrida
                
            print(f"  Corrida {corrida+1}/{num_ejecuciones} | Resultado: {mejor_tiempo_corrida:.3f} s | CPU: {tiempo_cpu:.2f} s")

        # CÁLCULO ESTADÍSTICO
        t_optimos = np.array(tiempos_optimos)
        print(f"\n REPORTE ESTADÍSTICO - {track_name}:")
        print(f" -> Mejor Tiempo:    {np.min(t_optimos):.3f} s")
        print(f" -> Peor Tiempo:     {np.max(t_optimos):.3f} s")
        print(f" -> Tiempo Promedio: {np.mean(t_optimos):.3f} s")
        print(f" -> Desviación Est.: {np.std(t_optimos):.4f} s")
        print(f" -> T. Computacional Medio: {np.mean(tiempos_computacionales):.2f} s por corrida")
        print(f" -> Mejora Máxima Lograda:  -{tiempo_base - np.min(t_optimos):.3f} s")
        
        # GUARDAR DATOS PARA EL COMPAÑERO DE VISUALIZACIÓN
        # Generamos la trayectoria suave final ganadora
        x_opt, y_opt, s_denso = generar_trayectoria(
            track_data['cx'], track_data['cy'], 
            track_data['nx'], track_data['ny'], 
            mejor_trayectoria_global
        )
        _, v_max_opt, _ = evaluar_fitness(x_opt, y_opt)
        
        # Empaquetamos la telemetría en un CSV para el Paso 6
        import pandas as pd
        df_telemetria = pd.DataFrame({
            'x': x_opt,
            'y': y_opt,
            'v_max_ms': v_max_opt
        })
        path_telemetria = out_dir / f"{track_name}_trayectoria_optima.csv"
        df_telemetria.to_csv(path_telemetria, index=False)
        print(f" Trayectoria óptima exportada en: {path_telemetria}\n")

if __name__ == "__main__":
    main()