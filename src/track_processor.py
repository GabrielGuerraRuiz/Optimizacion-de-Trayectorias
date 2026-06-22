import numpy as np

def procesar_pista(df):
    """
    Calcula los vectores normales para obtener las coordenadas
    exactas de los límites de la pista y permitir su visualización.
    Además, extrae características matemáticas (curvatura) para 
    reducir la dimensionalidad de la optimización.
    """
    cx = df["x_m"].to_numpy(dtype=float)
    cy = df["y_m"].to_numpy(dtype=float)
    w_right = df["w_tr_right_m"].to_numpy(dtype=float)
    w_left = df["w_tr_left_m"].to_numpy(dtype=float)

    # Vectores dirección (tangentes)
    dx = np.r_[cx[1:] - cx[:-1], cx[0] - cx[-1]]
    dy = np.r_[cy[1:] - cy[:-1], cy[0] - cy[-1]]

    norma = np.sqrt(dx**2 + dy**2)
    norma[norma == 0] = 1e-12 # Evita división entre cero

    tx = dx / norma
    ty = dy / norma

    # Vectores normales (apuntando al borde derecho)
    nx = ty
    ny = -tx

    # Coordenadas exactas de los bordes laterales
    borde_der_x = cx + nx * w_right
    borde_der_y = cy + ny * w_right
    borde_izq_x = cx - nx * w_left
    borde_izq_y = cy - ny * w_left

    # --- NUEVO: CÁLCULO DEL RADIO DE CURVATURA ---
    # Usamos la segunda derivada para encontrar las zonas complejas (curvas)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvatura = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-12)**1.5

    # --- NUEVO: COLOCACIÓN INTELIGENTE DE CHECKPOINTS ---
    indices_cp = [0]
    dist_acumulada = 0
    # Definimos como "curva fuerte" aquellas por encima de la media estadística
    umbral_curvatura = np.mean(curvatura) + 0.5 * np.std(curvatura)

    for i in range(1, len(cx)):
        dist_acumulada += norma[i]
        # Colocamos un punto de control si la curva es cerrada 
        # O si llevamos mucha distancia en línea recta (ej. 30 metros) para evitar derivas
        if curvatura[i] > umbral_curvatura or dist_acumulada > 30.0:
            indices_cp.append(i)
            dist_acumulada = 0

    # Garantizar que el circuito se cierre exactamente conectando con el último nodo
    if indices_cp[-1] != len(cx) - 1:
        indices_cp.append(len(cx) - 1)

    return {
        "cx": cx, "cy": cy,
        "nx": nx, "ny": ny,
        "w_left": w_left, "w_right": w_right,
        "borde_der_x": borde_der_x, "borde_der_y": borde_der_y,
        "borde_izq_x": borde_izq_x, "borde_izq_y": borde_izq_y,
        "indices_checkpoints": np.array(indices_cp), # Índices reducidos a optimizar
        "curvatura": curvatura # Útil para los cálculos automáticos de Zoom
    }