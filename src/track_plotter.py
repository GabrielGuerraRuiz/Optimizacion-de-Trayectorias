import matplotlib.pyplot as plt

def plot_track(track_data, track_name, tray_x=None, tray_y=None, ctrl_x=None, ctrl_y=None, save_path=None):
    """Grafica la pista, la trayectoria y los puntos de control del algoritmo."""
    if track_data is None: return

    plt.figure(figsize=(10, 8))
    
    # Geometría base
    plt.plot(track_data['cx'], track_data['cy'], 'r--', alpha=0.5, label='Línea Central')
    plt.plot(track_data['borde_der_x'], track_data['borde_der_y'], 'k-', label='Borde Derecho')
    plt.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], 'b-', label='Borde Izquierdo')
    
    # Trayectoria interpolada (Fenotipo)
    if tray_x is not None and tray_y is not None:
        plt.plot(tray_x, tray_y, color='magenta', linewidth=2, label='Trayectoria Candidata')
        
    # Puntos de Control (Genotipo)
    if ctrl_x is not None and ctrl_y is not None:
        plt.scatter(ctrl_x, ctrl_y, color='orange', s=35, zorder=5, label='Puntos de Control (Genes)')
    
    plt.title(f"Modelo de Trayectoria (Makima): {track_name}")
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()