import numpy as np
from scipy.interpolate import PchipInterpolator

def generar_trayectoria(cx, cy, nx, ny, desplazamientos):
    """
    Toma la línea central, los vectores normales y los desplazamientos 
    propuestos por una partícula del PSO para generar una trayectoria suave.
    """
    # 1. Aplicar los desplazamientos laterales a lo largo de las normales
    x_discreto = cx + nx * desplazamientos
    y_discreto = cy + ny * desplazamientos
    
    # 2. Calcular la distancia acumulada (S) para parametrizar la curva
    dx = np.r_[x_discreto[1:] - x_discreto[:-1], x_discreto[0] - x_discreto[-1]]
    dy = np.r_[y_discreto[1:] - y_discreto[:-1], y_discreto[0] - y_discreto[-1]]
    distancias = np.sqrt(dx**2 + dy**2)
    s_acumulado = np.concatenate(([0], np.cumsum(distancias)))
    
    # Para cerrar el circuito suavemente, duplicamos el primer punto al final
    x_cerrado = np.append(x_discreto, x_discreto[0])
    y_cerrado = np.append(y_discreto, y_discreto[0])
    
    # 3. Construir los interpoladores PCHIP (Alternativa en Python a Makima)
    interp_x = PchipInterpolator(s_acumulado, x_cerrado)
    interp_y = PchipInterpolator(s_acumulado, y_cerrado)
    
    # 4. Generar una trayectoria hiper-densa (ej. 5 veces más puntos) 
    # para un cálculo de derivadas numéricas mucho más estable
    s_denso = np.linspace(0, s_acumulado[-1], len(cx) * 5)
    x_continuo = interp_x(s_denso)
    y_continuo = interp_y(s_denso)
    
    return x_continuo, y_continuo, s_denso

def evaluar_fitness(x, y, mu=1.0, g=9.81, v_limite=88.8):
    """
    Evalúa el desempeño de la trayectoria usando el modelo de estabilidad lateral.
    Calcula velocidades máximas seguras basadas en curvatura y retorna el tiempo de vuelta.
    """
    # 1. Calcular derivadas numéricas de la trayectoria interpolada
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # 2. Calcular la curvatura local (\kappa)
    # Fórmula: \kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerador = np.abs((dx * ddy) - (dy * ddx))
    denominador = (dx**2 + dy**2)**(1.5)
    denominador = np.where(denominador == 0, 1e-12, denominador)
    kappa = numerador / denominador
    
    # 3. Calcular el radio de curvatura (r)
    # Evitamos división por cero en las rectas imponiendo una curvatura mínima
    kappa = np.where(kappa < 1e-6, 1e-6, kappa)
    radio = 1.0 / kappa
    
    # 4. Estimar velocidades máximas seguras [cite: 17]
    # Ecuación: vmax = sqrt(mu * g * r) [cite: 17]
    v_max = np.sqrt(mu * g * radio)
    v_max = np.clip(v_max, a_min=None, a_max=v_limite) # Restricción física del motor
    
    # 5. Estimar el tiempo total de vuelta [cite: 17]
    distancias_tramo = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    velocidades_tramo = v_max[:-1] # Usamos la velocidad al inicio de cada segmento
    
    # Evitamos dividir por cero si la velocidad llegara a ser nula
    velocidades_tramo = np.where(velocidades_tramo == 0, 1e-12, velocidades_tramo)
    tiempos_tramo = distancias_tramo / velocidades_tramo
    tiempo_total = np.sum(tiempos_tramo)
    
    return tiempo_total, v_max, radio