import numpy as np

def procesar_pista(df):
    """
    Calcula los vectores normales para obtener las coordenadas
    exactas de los límites de la pista y permitir su visualización.
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

    return {
        "cx": cx, "cy": cy,
        "nx": nx, "ny": ny,
        "w_left": w_left, "w_right": w_right,
        "borde_der_x": borde_der_x, "borde_der_y": borde_der_y,
        "borde_izq_x": borde_izq_x, "borde_izq_y": borde_izq_y
    }