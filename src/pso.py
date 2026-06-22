import numpy as np
from scipy.interpolate import CubicSpline
from src.evaluator import generar_trayectoria, evaluar_fitness

class EnjambrePSO:
    def __init__(self, track_data, config):
        """
        Inicializa el Enjambre PSO con hiperparámetros configurables.
        Emplea nodos de control (checkpoints) para reducir el espacio de búsqueda.
        """
        self.track = track_data
        self.num_p = config['num_particulas']
        self.num_iter = config['num_iteraciones']
        self.w = config['inercia_w']
        self.c1 = config['cognitivo_c1']
        self.c2 = config['social_c2']
        
        # --- REDUCCIÓN DE DIMENSIONALIDAD ---
        self.indices_cp = self.track['indices_checkpoints']
        self.dim = len(self.indices_cp)
        
        # LÍMITES DEL ESPACIO DE BÚSQUEDA
        self.limite_inf = -self.track['w_left'][self.indices_cp]
        self.limite_sup = self.track['w_right'][self.indices_cp]
        
        # 1. INICIALIZACIÓN DE LA MATRIZ DE POSICIONES
        self.posiciones = np.zeros((self.num_p, self.dim))
        
        # Inicialización aleatoria estrictamente en los nodos de control
        for i in range(1, self.num_p):
            self.posiciones[i] = np.random.uniform(self.limite_inf, self.limite_sup, self.dim)
            
        # ---> FIX: Garantizar que el inicio y el fin coincidan para el Spline Periódico
        self.posiciones[:, -1] = self.posiciones[:, 0]
            
        # El índice 0 se queda en la línea central (puros ceros)
        self.posiciones[0] = np.zeros(self.dim)
        
        # 2. INICIALIZACIÓN DE VELOCIDADES
        self.velocidades = np.zeros((self.num_p, self.dim))
        
        # 3. MEMORIA DEL ENJAMBRE
        self.pbest_pos = np.copy(self.posiciones)
        self.pbest_fit = np.full(self.num_p, np.inf)
        
        self.gbest_pos = None
        self.gbest_fit = np.inf
        
        self.historial_convergencia = []

    def optimizar(self, verbose=True):
        """
        Ejecuta el ciclo principal del PSO.
        """
        n_puntos_pista = len(self.track['cx'])
        puntos_totales = np.arange(n_puntos_pista)

        for t in range(self.num_iter):
            # A. FASE DE EVALUACIÓN
            for i in range(self.num_p):
                despl_cp = self.posiciones[i]
                
                # --- INTERPOLACIÓN MATEMÁTICA ---
                cs = CubicSpline(self.indices_cp, despl_cp, bc_type='periodic')
                desplazamientos_completos = cs(puntos_totales)
                
                # Generamos la trayectoria visual completa
                x_suave, y_suave, _ = generar_trayectoria(
                    self.track['cx'], self.track['cy'],
                    self.track['nx'], self.track['ny'],
                    desplazamientos_completos
                )
                
                # Evaluamos el tiempo de vuelta
                tiempo, _, _ = evaluar_fitness(x_suave, y_suave)
                
                # B. ACTUALIZACIÓN DE MEMORIA INDIVIDUAL (pBest)
                if tiempo < self.pbest_fit[i]:
                    self.pbest_fit[i] = tiempo
                    self.pbest_pos[i] = np.copy(despl_cp)
                    
                    # C. ACTUALIZACIÓN DE MEMORIA GLOBAL (gBest)
                    if tiempo < self.gbest_fit:
                        self.gbest_fit = tiempo
                        self.gbest_pos = np.copy(despl_cp)
            
            # Registro de telemetría por iteración
            self.historial_convergencia.append(self.gbest_fit)
            if verbose:
                print(f"  Iteración {t+1}/{self.num_iter} | Mejor Tiempo: {self.gbest_fit:.3f} s")

            # D. FASE DE MOVIMIENTO
            r1 = np.random.rand(self.num_p, self.dim)
            r2 = np.random.rand(self.num_p, self.dim)
            
            vel_cognitiva = self.c1 * r1 * (self.pbest_pos - self.posiciones)
            vel_social = self.c2 * r2 * (self.gbest_pos - self.posiciones)
            
            self.velocidades = (self.w * self.velocidades) + vel_cognitiva + vel_social
            self.posiciones = self.posiciones + self.velocidades
            
            # E. RESTRICCIÓN DE MUROS
            self.posiciones = np.clip(self.posiciones, self.limite_inf, self.limite_sup)
            
            # ---> FIX: Mantener la condición periódica tras la actualización de las partículas
            self.posiciones[:, -1] = self.posiciones[:, 0]
            
        # Al finalizar, generamos el array completo
        cs_final = CubicSpline(self.indices_cp, self.gbest_pos, bc_type='periodic')
        mejor_trayectoria_comprimida = cs_final(puntos_totales)
            
        return mejor_trayectoria_comprimida, self.gbest_fit, self.historial_convergencia