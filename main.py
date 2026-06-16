from src.track_loader import load_track_data
from src.track_plotter import plot_centerline

def main():
    # 1. Definir las rutas
    sakhir_path = "data/sakhir.csv"
    montreal_path = "data/montreal.csv"
    
    # 2. Cargar los datos
    print("Iniciando carga de pistas...")
    sakhir_data = load_track_data(sakhir_path)
    
    # 3. Graficar (Paso 1 del proyecto)
    if sakhir_data is not None:
        plot_centerline(sakhir_data, "Circuito de Sakhir")

if __name__ == "__main__":
    main()