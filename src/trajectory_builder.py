import numpy as np
from scipy.interpolate import Akima1DInterpolator

def calcular_arco_cerrado(cx, cy):
    """Calcula la distancia acumulada 's' a lo largo del circuito."""
    dx = np.r_[cx[1:] - cx[:-1], cx[0] - cx[-1]]
    dy = np.r_[cy[1:] - cy[:-1], cy[0] - cy[-1]]
    ds = np.sqrt(dx**2 + dy**2)
    s = np.r_[0.0, np.cumsum(ds[:-1])]
    longitud_total = float(np.sum(ds))
    return s, longitud_total

def elegir_puntos_control(n_puntos, n_controles=80):
    """Selecciona los índices de los puntos que actuarán como los 'Genes' del GA."""
    return np.linspace(0, n_puntos, n_controles, endpoint=False, dtype=int)

def obtener_limites_desplazamiento(track_data, margen=0.5):
    """Calcula los límites seguros de desplazamiento lateral (con margen de error)."""
    lb = -track_data["w_left"] + margen
    ub = track_data["w_right"] - margen
    return lb, ub

def generar_individuo_aleatorio(lb, ub, indices_control, escala=0.8):
    """Genera un vector aleatorio de desplazamientos laterales (Genotipo)."""
    lb_c = lb[indices_control]
    ub_c = ub[indices_control]
    centro = 0.5 * (lb_c + ub_c)
    radio = 0.5 * (ub_c - lb_c) * escala
    return np.random.uniform(centro - radio, centro + radio)

def interpolar_makima(s, longitud_total, indices_control, u_control, lb, ub):
    """Expande los genes de control a todos los puntos de la pista usando Makima."""
    s_control = s[indices_control]
    
    # Truco de expansión para asegurar continuidad perfecta en circuito cerrado
    k = min(4, len(indices_control))
    s_ext = np.r_[s_control[-k:] - longitud_total, s_control, s_control[:k] + longitud_total]
    u_ext = np.r_[u_control[-k:], u_control, u_control[:k]]
    
    interpolador = Akima1DInterpolator(s_ext, u_ext, method="makima")
    u_full = interpolador(s)
    
    # Recorte de seguridad para evitar invasión de límites
    return np.clip(u_full, lb, ub)

def construir_coordenadas_trayectoria(track_data, u_full):
    """Proyecta los desplazamientos matemáticos en coordenadas espaciales X, Y."""
    tray_x = track_data['cx'] + (u_full * track_data['nx'])
    tray_y = track_data['cy'] + (u_full * track_data['ny'])
    return tray_x, tray_y