import matplotlib.pyplot as plt

def plot_track(track_data, track_name, save_path=None):
    """
    Visualiza gráficamente la línea central y los límites de la pista.
    """
    if track_data is None:
        return

    plt.figure(figsize=(10, 8))
    
    # Dibujo de la pista
    plt.plot(track_data['cx'], track_data['cy'], 'r--', label='Línea Central')
    plt.plot(track_data['borde_der_x'], track_data['borde_der_y'], 'k-', label='Borde Derecho')
    plt.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], 'b-', label='Borde Izquierdo')
    
    # Formato gráfico
    plt.title(f"Modelado Geométrico: {track_name}")
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.axis('equal') # CRÍTICO: Mantiene la proporción real de la pista
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()