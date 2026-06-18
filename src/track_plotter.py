import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def plot_track(track_data, track_name, save_path=None):
    """
    Visualiza gráficamente la línea central y los límites de la pista.
    """
    if track_data is None:
        return

    plt.figure(figsize=(10, 8))
    plt.plot(track_data['cx'], track_data['cy'], 'r--', label='Línea Central')
    plt.plot(track_data['borde_der_x'], track_data['borde_der_y'], 'k-', label='Borde Derecho')
    plt.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], 'b-', label='Borde Izquierdo')
    plt.title(f"Modelado Geométrico: {track_name}")
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


def plot_telemetry_panel(track_data, track_name, best_traj, convergence_history, baseline_stats, optimized_stats, save_path=None):
    """
    Panel dinámico de telemetría para PSO:
    1) Track heatmap con velocidad instantánea (en km/h).
    2) Curva de convergencia del enjambre.
    3) Panel comparativo baseline vs optimización.
    """
    if track_data is None or best_traj is None:
        return

    x_opt = np.asarray(best_traj['x'])
    y_opt = np.asarray(best_traj['y'])
    
    # [CORRECCIÓN]: Extraemos m/s y lo convertimos inmediatamente a km/h para la gráfica
    v_max_opt_ms = np.asarray(best_traj['v_max_ms'])
    v_max_opt_kmh = v_max_opt_ms * 3.6 

    if len(x_opt) == 0 or len(y_opt) == 0:
        return

    conv = np.asarray(convergence_history, dtype=float) if convergence_history is not None else np.array([])

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[2, 1], wspace=0.25, hspace=0.28)

    ax_track = fig.add_subplot(gs[:, 0])
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_track.plot(track_data['borde_der_x'], track_data['borde_der_y'], color='dimgray', linewidth=1.6, label='Borde Derecho')
    ax_track.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], color='gray', linewidth=1.6, label='Borde Izquierdo')
    ax_track.plot(track_data['cx'], track_data['cy'], color='black', linestyle='--', linewidth=1.0, alpha=0.65, label='Línea Central')

    # [CORRECCIÓN]: Usamos la variable v_max_opt_kmh para los colores del mapa de calor
    sc = ax_track.scatter(x_opt, y_opt, c=v_max_opt_kmh, cmap='coolwarm', s=18, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax_track, fraction=0.046, pad=0.04)
    
    # [CORRECCIÓN]: Cambiamos la etiqueta a km/h
    cbar.set_label('Velocidad instantánea (km/h)', rotation=90)

    ax_track.set_title(f"{track_name} - Mapa de Calor de Velocidad")
    ax_track.set_xlabel('Coordenada X (m)')
    ax_track.set_ylabel('Coordenada Y (m)')
    ax_track.axis('equal')
    ax_track.grid(True, linestyle='--', alpha=0.25)
    ax_track.legend(loc='best')

    if conv.size > 0:
        ax_conv.plot(np.arange(1, len(conv) + 1), conv, color='#2c7fb8', linewidth=2.0, marker='o', markersize=4)
    ax_conv.set_title('Convergencia del Enjambre')
    ax_conv.set_xlabel('Iteraciones')
    ax_conv.set_ylabel('Mejor tiempo de vuelta g_best (s)')
    ax_conv.grid(True, linestyle='--', alpha=0.35)

    # Las estadísticas ya vienen correctamente calculadas desde main.py / GUI
    tiempo_base, vel_base, _ = baseline_stats
    tiempo_opt, vel_opt, _ = optimized_stats
    mejora = tiempo_base - tiempo_opt

    ax_text.axis('off')
    panel = (
        "Comparativa Baseline vs Optimización\n"
        "-----------------------------------\n"
        f"Tiempo base (Línea Central): {tiempo_base:.3f} s\n"
        f"Tiempo optimizado (PSO):      {tiempo_opt:.3f} s\n"
        f"Vel. promedio base:           {vel_base:.2f} km/h\n"
        f"Vel. promedio optimizada:     {vel_opt:.2f} km/h\n"
        f"Mejora neta Δt:               {mejora:.3f} s\n"
    )
    ax_text.text(
        0.02,
        0.95,
        panel,
        transform=ax_text.transAxes,
        va='top',
        ha='left',
        fontsize=12,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray', alpha=0.95)
    )

    fig.suptitle(f'Panel de Telemetría Dinámico - {track_name}', fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()