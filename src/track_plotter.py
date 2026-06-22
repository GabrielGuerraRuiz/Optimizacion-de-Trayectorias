import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
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
    Panel estático de telemetría para PSO.
    """
    if track_data is None or best_traj is None:
        return

    x_opt = np.asarray(best_traj['x'])
    y_opt = np.asarray(best_traj['y'])
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

    sc = ax_track.scatter(x_opt, y_opt, c=v_max_opt_kmh, cmap='coolwarm', s=18, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax_track, fraction=0.046, pad=0.04)
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
        0.02, 0.95, panel, transform=ax_text.transAxes,
        va='top', ha='left', fontsize=12, family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray', alpha=0.95)
    )

    fig.suptitle(f'Panel de Telemetría Dinámico - {track_name}', fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def animar_trayectoria_con_zoom(track_data, best_traj, convergence_history=None, baseline_stats=None, optimized_stats=None):
    """
    Animación interactiva con tres secciones:
    1. Izquierda: Pista con coche en movimiento y zoom automático.
    2. Derecha Arriba: Gráfica de convergencia dibujándose dinámicamente.
    3. Derecha Abajo: Cuadro estático comparativo de mejoras.
    """
    if track_data is None or best_traj is None:
        return

    x_opt = np.asarray(best_traj['x'])
    y_opt = np.asarray(best_traj['y'])

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("Replay Dinámico - Convergencia y Resultados")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[2, 1], wspace=0.2, hspace=0.3)

    # --- PANEL IZQUIERDO: PISTA Y ZOOM ---
    ax_track = fig.add_subplot(gs[:, 0])
    ax_track.plot(track_data['borde_der_x'], track_data['borde_der_y'], color='dimgray', linewidth=1.5)
    ax_track.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], color='gray', linewidth=1.5)
    ax_track.plot(track_data['cx'], track_data['cy'], 'r--', alpha=0.35, label='Línea Central')
    ax_track.axis('equal')
    ax_track.grid(True, linestyle=':', alpha=0.5)
    ax_track.set_title("Animación de Trayectoria Óptima con Zoom", fontsize=12)

    estela, = ax_track.plot([], [], 'b-', linewidth=2, alpha=0.75, label='Trayectoria PSO')
    coche, = ax_track.plot([], [], marker='s', color='lime', markersize=6, markeredgecolor='black', zorder=5)
    ax_track.legend(loc='upper left')

    # Zoom
    idx_zoom = np.argmax(track_data['curvatura']) if 'curvatura' in track_data else len(x_opt) // 2
    xz, yz = track_data['cx'][idx_zoom], track_data['cy'][idx_zoom]

    axins = zoomed_inset_axes(ax_track, zoom=3.0, loc='upper right')
    axins.plot(track_data['borde_der_x'], track_data['borde_der_y'], color='dimgray', linewidth=1.8)
    axins.plot(track_data['borde_izq_x'], track_data['borde_izq_y'], color='gray', linewidth=1.8)
    axins.plot(track_data['cx'], track_data['cy'], 'r--', alpha=0.4)
    
    estela_ins, = axins.plot([], [], 'b-', linewidth=2.5)
    coche_ins, = axins.plot([], [], marker='s', color='lime', markersize=9, markeredgecolor='black', zorder=5)

    rango_vista = 30
    axins.set_xlim(xz - rango_vista, xz + rango_vista)
    axins.set_ylim(yz - rango_vista, yz + rango_vista)
    mark_inset(ax_track, axins, loc1=2, loc2=4, fc="none", ec="0.4", linestyle='--')
    axins.axis('off')
    axins.set_title("Ápice Crítico", fontsize=9, pad=4)

    # --- PANEL DERECHO SUPERIOR: CONVERGENCIA ---
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_conv.set_title("Convergencia del Enjambre (PSO)", fontsize=12)
    ax_conv.set_xlabel("Iteraciones")
    ax_conv.set_ylabel("Tiempo de Vuelta g_best (s)")
    ax_conv.grid(True, linestyle='--', alpha=0.5)

    conv = np.asarray(convergence_history, dtype=float) if convergence_history is not None else np.array([])
    linea_conv, = ax_conv.plot([], [], color='#2c7fb8', linewidth=2.5, marker='o', markersize=4)

    if conv.size > 0:
        ax_conv.set_xlim(1, len(conv))
        ax_conv.set_ylim(np.min(conv) - 0.5, np.max(conv) + 0.5)
        # Sombra de fondo para indicar hacia dónde va la curva
        ax_conv.plot(np.arange(1, len(conv) + 1), conv, color='lightgray', linewidth=1.5, zorder=1)

    # --- PANEL DERECHO INFERIOR: COMPARATIVA ---
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')

    if baseline_stats and optimized_stats:
        tiempo_base, vel_base, _ = baseline_stats
        tiempo_opt, vel_opt, _ = optimized_stats
        mejora = tiempo_base - tiempo_opt

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
            0.05, 0.9, panel, transform=ax_text.transAxes,
            va='top', ha='left', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray', alpha=0.95)
        )

    # --- FUNCIONES DE ANIMACIÓN ---
    def init():
        estela.set_data([], [])
        coche.set_data([], [])
        estela_ins.set_data([], [])
        coche_ins.set_data([], [])
        linea_conv.set_data([], [])
        return estela, coche, estela_ins, coche_ins, linea_conv

    def update(frame):
        # Actualizar coche en pista
        estela.set_data(x_opt[:frame], y_opt[:frame])
        coche.set_data([x_opt[frame]], [y_opt[frame]])
        
        # Actualizar coche en zoom
        estela_ins.set_data(x_opt[:frame], y_opt[:frame])
        coche_ins.set_data([x_opt[frame]], [y_opt[frame]])
        
        # Sincronizar el dibujo de la curva de convergencia con el avance del coche
        if conv.size > 0:
            progreso = frame / len(x_opt)
            idx_iter = max(1, int(progreso * len(conv)))
            linea_conv.set_data(np.arange(1, idx_iter + 1), conv[:idx_iter])
            
        return estela, coche, estela_ins, coche_ins, linea_conv

    submuestreo = max(1, len(x_opt) // 350) 
    indices_frames = np.arange(0, len(x_opt), submuestreo)
    if indices_frames[-1] != len(x_opt) - 1:
        indices_frames = np.append(indices_frames, len(x_opt) - 1)
    
    ani = animation.FuncAnimation(
        fig, update, frames=indices_frames, 
        init_func=init, blit=True, interval=25, repeat=True
    )
    plt.show(block=True)