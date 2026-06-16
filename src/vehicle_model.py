import numpy as np

def calcular_curvatura(x, y):
    """
    Calcula la curvatura local (kappa) en cada punto de la trayectoria
    utilizando derivadas de diferencias finitas centrales.
    """
    # Primera derivada (Velocidad geométrica)
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Segunda derivada (Aceleración geométrica)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Fórmula de curvatura para curvas paramétricas bidimensionales
    numerador = (dx * ddy) - (dy * ddx)
    denominador = (dx**2 + dy**2)**(1.5)
    
    # Prevenir división por cero en caso de puntos duplicados
    denominador = np.where(denominador == 0, 1e-12, denominador)
    
    kappa = np.abs(numerador / denominador)
    return kappa

def evaluar_tiempo_vuelta(x, y, mu=1.0, g=9.81, v_limite=88.8):
    """
    Evalúa físicamente una trayectoria (x, y).
    mu: Coeficiente de fricción (1.0 es típico para neumáticos de carrera).
    g: Gravedad (9.81 m/s^2).
    v_limite: Límite físico del auto en m/s (88.8 m/s = ~320 km/h).
    Retorna el tiempo total de la vuelta y el arreglo de velocidades.
    """
    # 1. Distancias entre puntos consecutivos (Delta s)
    # Usamos np.r_ como tu compañero para cerrar la pista (del último al primer punto)
    dx = np.r_[x[1:] - x[:-1], x[0] - x[-1]]
    dy = np.r_[y[1:] - y[:-1], y[0] - y[-1]]
    ds = np.sqrt(dx**2 + dy**2)
    ds = np.where(ds == 0, 1e-12, ds) # Seguridad
    
    # 2. Curvatura y Radio
    kappa = calcular_curvatura(x, y)
    
    # En rectas, la curvatura es casi cero, lo que da un radio infinito.
    # Limitamos kappa mínimo para evitar división por cero.
    kappa = np.where(kappa < 1e-6, 1e-6, kappa)
    radio = 1.0 / kappa
    
    # 3. Velocidad Máxima Segura (Modelo simplificado del proyecto)
    v_max = np.sqrt(mu * g * radio)
    
    # La matemática pura de la fórmula daría velocidades infinitas en las rectas.
    # Aplicamos un techo realista para la velocidad máxima del vehículo.
    v_max = np.clip(v_max, a_min=None, a_max=v_limite)
    
    # 4. Tiempo de vuelta
    tiempos_tramo = ds / v_max
    tiempo_total = np.sum(tiempos_tramo)
    
    return tiempo_total, v_max