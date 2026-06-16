import pandas as pd

def load_track_data(filepath):
    """
    Carga la línea central y límites laterales desde el CSV.
    """
    try:
        df = pd.read_csv(filepath)
        # Renombramos la primera columna para manejarla más fácil
        df = df.rename(columns={df.columns[0]: "x_m"})
        
        columnas_necesarias = {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"}
        if not columnas_necesarias.issubset(df.columns):
            raise ValueError(f"Faltan columnas esperadas en {filepath}")
            
        return df
        
    except Exception as e:
        print(f"Error al cargar la pista: {e}")
        return None