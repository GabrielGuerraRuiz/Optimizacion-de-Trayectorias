import pandas as pd

def load_track_data(filepath):
    """
    Carga los datos crudos de la pista desde un CSV.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Mapeo exacto basado en el diagnóstico de la consola
        track_data = {
            'x': df['# x_m'].values,           
            'y': df['y_m'].values,           
            'w_right': df['w_tr_right_m'].values,
            'w_left': df['w_tr_left_m'].values  
        }
        
        print(f"Pista cargada con éxito: {len(track_data['x'])} puntos.")
        return track_data
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {filepath}")
        return None
    except KeyError as e:
        print(f"Error crítico de columnas: falta la columna {e}")
        return None