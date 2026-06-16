import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.evaluator import generar_trayectoria, evaluar_fitness

class EnjambrePSO:
    def __init__(self, track_data, config):
        """
        Inicializa el Enjambre PSO con hiperparámetros configurables.
        """
        self.track = track_data
        self.num_p = config['num_particulas']
        self.num_iter = config['num_iteraciones']
        self.w = config['inercia_w']
        self.c1 = config['cognitivo_c1']
        self.c2 = config['social_c2']
        
        # Dimensiones: Una partícula tiene tantos valores como puntos en la línea central
        self.dim = len(self.track['cx'])
        
        # LÍMITES DEL ESPACIO DE BÚSQUEDA
        # Un desplazamiento negativo mueve el auto a la izquierda, uno positivo a la derecha.
        # Por lo tanto, el límite inferior es -w_left y el superior es +w_right.
        self.limite_inf = -self.track['w_left']
        self.limite_sup = self.track['w_right']
        
        # 1. INICIALIZACIÓN DE LA MATRIZ DE POSICIONES (num_p x dim)
        self.posiciones = np.zeros((self.num_p, self.dim))
        
        # Generamos ruido y lo suavizamos para crear curvas base realistas
        for i in range(1, self.num_p): # Saltamos el 0
            ruido = np.random.uniform(self.limite_inf, self.limite_sup, self.dim)
            # mode='wrap' es vital porque la pista es un circuito cerrado (el final conecta con el inicio)
            self.posiciones[i] = gaussian_filter1d(ruido, sigma=15.0, mode='wrap')
            
        # El índice 0 se queda como la línea central (puros ceros)
        self.posiciones[0] = np.zeros(self.dim)
        
        # 2. INICIALIZACIÓN DE VELOCIDADES
        self.velocidades = np.zeros((self.num_p, self.dim))
        
        # 3. MEMORIA DEL ENJAMBRE
        self.pbest_pos = np.copy(self.posiciones)  # Mejor posición histórica de CADA partícula
        self.pbest_fit = np.full(self.num_p, np.inf) # Tiempos de vuelta iniciales (infinito)
        
        self.gbest_pos = None     # Mejor posición global
        self.gbest_fit = np.inf   # Mejor tiempo de vuelta global
        
        # Telemetría para el compañero de visualización
        self.historial_convergencia = []

    def optimizar(self, verbose=True):
        """
        Ejecuta el ciclo principal del PSO.
        """
        for t in range(self.num_iter):
            # A. FASE DE EVALUACIÓN
            for i in range(self.num_p):
                # Extraemos los desplazamientos de la partícula 'i'
                desplazamientos = self.posiciones[i]
                
                # Generamos la trayectoria suave
                x_suave, y_suave, _ = generar_trayectoria(
                    self.track['cx'], self.track['cy'],
                    self.track['nx'], self.track['ny'],
                    desplazamientos
                )
                
                # Evaluamos el tiempo de vuelta
                tiempo, _, _ = evaluar_fitness(x_suave, y_suave)
                
                # B. ACTUALIZACIÓN DE MEMORIA INDIVIDUAL (pBest)
                if tiempo < self.pbest_fit[i]:
                    self.pbest_fit[i] = tiempo
                    self.pbest_pos[i] = np.copy(desplazamientos)
                    
                    # C. ACTUALIZACIÓN DE MEMORIA GLOBAL (gBest)
                    if tiempo < self.gbest_fit:
                        self.gbest_fit = tiempo
                        self.gbest_pos = np.copy(desplazamientos)
            
            # Registro de telemetría por iteración
            self.historial_convergencia.append(self.gbest_fit)
            if verbose:
                print(f"  Iteración {t+1}/{self.num_iter} | Mejor Tiempo: {self.gbest_fit:.3f} s")

            # D. FASE DE MOVIMIENTO (Actualización Vectorizada)
            r1 = np.random.rand(self.num_p, self.dim)
            r2 = np.random.rand(self.num_p, self.dim)
            
            vel_cognitiva = self.c1 * r1 * (self.pbest_pos - self.posiciones)
            vel_social = self.c2 * r2 * (self.gbest_pos - self.posiciones)
            
            self.velocidades = (self.w * self.velocidades) + vel_cognitiva + vel_social
            self.posiciones = self.posiciones + self.velocidades
            
            # ---> NUEVO: SUAVIZADO DEL ENJAMBRE <---
            # Obligamos a que la nueva posición sea una curva matemáticamente continua
            for i in range(self.num_p):
                self.posiciones[i] = gaussian_filter1d(self.posiciones[i], sigma=5.0, mode='wrap')
            
            # E. RESTRICCIÓN DE MUROS (Clamping)
            # Si una partícula cruza el muro tras la actualización, la forzamos al límite de la pista
            self.posiciones = np.clip(self.posiciones, self.limite_inf, self.limite_sup)
            
        return self.gbest_pos, self.gbest_fit, self.historial_convergencia