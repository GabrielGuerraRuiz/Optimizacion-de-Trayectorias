import os
from pathlib import Path
from src.track_loader import load_track_data
from src.track_processor import procesar_pista
from src.track_plotter import plot_track

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

    print("Se han procesado las pistas seleccionadas")

if __name__ == "__main__":
    main()