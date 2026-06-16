import numpy as np
from pathlib import Path
from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.track_plotter import plot_track
import src.trajectory_builder as tb

def main():
    tracks = ["MexicoCity", "Montreal", "Monza", "Oschersleben", "Sakhir"]
    data_dir = Path("data")
    fig_dir = Path("resultados/figuras")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Iniciando generación de trayectorias...\n")

    for track_name in tracks:
        file_path = data_dir / f"{track_name}.csv"
        df_raw = load_track_data(file_path)
        if df_raw is None: continue
            
        track_data = procesar_pista(df_raw)
        
        # --- PARTE 2: LÓGICA MAKIMA ADAPTADA ---
        n_puntos = len(track_data['cx'])
        
        # Geometría paramétrica y límites
        s, longitud_total = tb.calcular_arco_cerrado(track_data['cx'], track_data['cy'])
        lb, ub = tb.obtener_limites_desplazamiento(track_data, margen=0.5)
        
        # 1. Definir los genes
        indices_control = tb.elegir_puntos_control(n_puntos, n_controles=80)
        
        # 2. Generar el individuo base (los desplazamientos)
        u_control = tb.generar_individuo_aleatorio(lb, ub, indices_control)
        
        # 3. Interpolar la curva fluida con Makima
        u_full = tb.interpolar_makima(s, longitud_total, indices_control, u_control, lb, ub)
        
        # 4. Obtener las coordenadas físicas reales y separar puntos de control
        tray_x, tray_y = tb.construir_coordenadas_trayectoria(track_data, u_full)
        ctrl_x = tray_x[indices_control]
        ctrl_y = tray_y[indices_control]
        
        # Exportar resultado visual
        save_fig_path = fig_dir / f"{track_name}_trayectoria.png"
        plot_track(track_data, track_name, tray_x, tray_y, ctrl_x, ctrl_y, save_path=save_fig_path)
        print(f" -> Trayectoria robusta generada: {save_fig_path}")

if __name__ == "__main__":
    main()