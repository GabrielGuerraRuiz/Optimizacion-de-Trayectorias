import matplotlib.pyplot as plt

def plot_centerline(track_data, track_name="Pista"):
    """
    Grafica la línea central de la pista.
    """
    if track_data is None:
        return

    plt.figure(figsize=(10, 6))
    
    # Graficar línea central
    plt.plot(track_data['x'], track_data['y'], label='Línea Central', color='blue', linewidth=2)
    
    # Formato de la gráfica
    plt.title(f"Modelado Geométrico: {track_name}")
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.axis('equal') # CRÍTICO: Mantiene la proporción real 1:1 de la pista
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()