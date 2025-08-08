# Importing required libraries
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import matplotlib.pyplot as plt


class grid:
    def __init__(self, nodes, lines, pros):
        self.nodes = self.add_nodes(nodes)                                      
        self.lines = self.add_lines(lines, self.nodes)  
        self.pros = self.add_pros(pros, self.nodes)  
        self.n = len(self.nodes)
        self.m = len(self.lines)
        self.x_size = self.n+(2*self.m)-1
        self.initialize()
        
    def add_nodes(self, nodes):        
        nodes_list = list()
        for item in nodes:
            nodes_list.append(node(item['id'], item['slack']))
        return nodes_list
        
    def add_lines(self, lines, nodes):        
        lines_list = list()
        for item in lines:
            lines_list.append(line(item['id'], item['From'], item['To'], item['R'], item['X'], nodes))
        return lines_list
        
    def add_pros(self, pros, nodes):        
        pros_list = list()
        for item in pros:
            pros_list.append(prosumer(item['id'], item['Node'], item['P'], item['Q'], nodes))
        return pros_list
    
    def initialize(self):        
        n_aux = 0
        while n_aux < self.n:
            self.nodes[n_aux].index = n_aux -1
            n_aux += 1        
        n_aux2 = 0    
        while n_aux2 < self.m:   
            self.lines[n_aux2].index.append(n_aux - 1)
            n_aux += 1
            n_aux2 += 1        
        n_aux3 = 0    
        while n_aux3 < self.m:   
            self.lines[n_aux3].index.append(n_aux - 1)
            n_aux += 1
            n_aux3 += 1       
        self.x = np.zeros(self.x_size)
        self.x[:self.n - 1] = 1
        
    def obtain_A(self):        
        matrizA = np.zeros(((2*self.n)-2, (self.n+2*self.m)-1), dtype=float)        
        n_aux = 0
        for i, node in enumerate(self.nodes[1:]):
            matrizA[2*i, n_aux] = np.sum([line.G for line in node.lines])
            matrizA[2*i+1, n_aux] = np.sum([line.B for line in node.lines])
            n_aux += 1
            
            for j, line in enumerate(node.lines):
                if node == line.nodes[0]:
                    matrizA[2*i, line.index[0]] = -line.G
                    matrizA[2*i, line.index[1]] = -line.B
                    matrizA[2*i+1, line.index[0]] = -line.B
                    matrizA[2*i+1, line.index[1]] = line.G
                else:
                    matrizA[2*i, line.index[0]] = -line.G
                    matrizA[2*i, line.index[1]] = line.B
                    matrizA[2*i+1, line.index[0]] = -line.B
                    matrizA[2*i+1, line.index[1]] = -line.G            
        self.A = matrizA
        
    def ineq(self, x):        
        rest = []
        for line in self.lines:
            rest.append(line.ineq(x))          
        return rest
    
    def ineq_jac(self, X):
        jacobian = []
        for line in self.lines:
            jacobian.append(line.ineq_jac(X))
        return np.array(jacobian)
    
    def compute_line_currents(self):
        current = []
        for line in self.lines:
            current.append(line.current())
        return current    
    
    def compute_node_currents(self):
        current = []
        for pros in self.pros:
            current.append(pros.current()) 
        return current    
    
    def check(self, tolerancia = 1e-2):
        check = []
        for node in self.nodes[1:]:
            total_intens = 0 + 0j
            for line in node.lines:
                if line.nodes[0] == node:
                    total_intens -= line.current()  
                elif line.nodes[1] == node:
                    total_intens += line.current()    
            for pros in node.pros:
                total_intens += pros.current() 
            if abs(total_intens) < tolerancia:
                check.append(True)
            else: 
                check.append(total_intens) 
        return check       
                
    def solve_iterative(self):       
        # Preparar las matrices que vamos a necesitar
        
        self.obtain_A()  # Matriz de igualdades A
        self.obtain_B()  # Vector de igualdades B
        self.obtain_f()  # Función objetivo f
        
        # Algoritmo iterativo: Descent-primal ascent-dual
        
        
        # Configuración general
        max_outer_iter = 10        # iteraciones externas para ajustar rho
        rho = 1                    # penalización inicial
        alpha = 0.00001            # paso inicial
        tol = 1e-6
        max_inner_iter = 1000
        
        for outer in range(max_outer_iter):
            print(f"\n🔁 Iteración externa {outer+1} — rho = {rho}")
        
        # Inicializar las variables
        
            x = self.x.copy() # x(k=0) Vector de decisioón (copiamos para no modificar el original directamente)
        #A = self.A
        #b = self.B
        #f = self.f
            mu = np.zeros(len(self.ineq(x)))  # Multiplicadores de las restricciones no lineales (uno por línea)
            lam = np.zeros(len(self.B))       # Multiplicadores de las restricciones lineales (2 por nodo excepto el Slack)
            s = np.zeros(len(mu))             # Variables de holgura, mismas dimensiones que mu
       # alpha = 0.000001                       # Tasa de aprendizaje
       # rho = 100                           # Parámetro del Lagrangiano aumentado
       # max_iter = 1000                   # Número máximo de iteraciones
       # tol = 1e-6                        # Tolerancia para parar
       # sol = 0
        
            print("Iteración | Error gradiente | Error igualdad")

            for k in range(max_inner_iter):
            # Paso 3: Calculamos g(x), restricciones no lineales
                gx = self.ineq(x)
                gx = np.clip(gx, -1e10, 1e10)  # Evitar explosiones con penalizacion gorda
                
            # Paso 4: Actualizamos s: variable de holgura
               
                #s = np.maximum(0, -mu / rho - gx)  # Si alguna componente es negativa, se queda en 0
                s = np.maximum(0, -gx - mu / rho) #modifico la funcion S por una equivalente por si es error de python
                
                # 🧪 Depuración cada 100 iteraciones
                if k % 100 == 0:
                    print(f"Iter {k:4d} | gx[:3] = {gx[:3]}, s[:3] = {s[:3]}, mu[:3] = {mu[:3]}")
                
                
            # Paso 5: Calculamos el gradiente del Lagrangiano aumentado
                grad_f = self.f.flatten()                      # f es una matriz (1, n), la pasamos a vector (n,)
                jac_g = self.ineq_jac(x)                       # Jacobiano de g(x)
                
                penalty = mu + rho * (gx + s)
                penalty = np.clip(penalty, -1e5, 1e5)  # para estabilidad
                
                # penalización directa adicional:
                penalty_extra = 10 * np.maximum(0, gx + s)
                penalty += penalty_extra
                
                grad_L = grad_f + self.A.T @ lam + jac_g.T @ penalty
            
            # Paso 6: Actualizamos x con descenso del gradiente
                x = x - alpha * grad_L
                x = np.clip(x, -100, 100)  # proteger estabilidad
                
            # Errores para control de convergencia
                error_grad = np.linalg.norm(grad_L)
                error_eq = np.linalg.norm(self.A @ x - self.B)
            
           

            # Condición de parada
                if error_grad < tol and error_eq < tol:
                    print(f"✅ Convergencia alcanzada en {k} pasos internos.")
                    break
       
            # Paso 7: Actualizamos los multiplicadores duales
                lam = lam + rho * (self.A @ x - self.B)
                mu = mu + rho * (gx + s) 
               
            # Paso 8: Limitar el crecimiento de mu y lam
                lam = np.clip(lam, -1e5, 1e5)
                mu = np.clip(mu, -1e5, 1e5)

             # Verificar restricciones cónicas
            violations = [line.ineq(x) > 0 for line in self.lines]
            if not any(violations):
                print("✅ Restricciones cónicas satisfechas.")
                break
            else:
                print("⚠️ Aún hay restricciones violadas. Aumentando rho...")
                rho *= 10
                alpha *= 0.5  # reducir paso por seguridad

        self.x = x
        for index, node in enumerate(self.nodes[1:]):
                node.Ckk = x[index]

        ##################################################
        
        return x
        
    
    def solve_pf(self):        
        self.obtain_A()
        self.obtain_B()
        self.obtain_f()        
        
        lc = LinearConstraint(self.A, self.B, self.B)
        nlc = NonlinearConstraint(self.ineq, -np.inf, 0)
        fo = lambda x: self.f.dot(x)
        sol = minimize(fo, self.x, constraints=(lc, nlc))        
        print(sol.message)
       
        for index, node in enumerate(self.nodes[1:]):
            node.Ckk = sol.x[index]  
            
        self.x = sol.x
        return sol
        
    
    def obtain_B(self):        
        matrizB = np.zeros(2*self.n-2, dtype=float)        
        for i, node in enumerate(self.nodes[1:]):           
            for x in node.pros:
                matrizB[2*i] += x.P
                matrizB[2*i+1] += x.Q             
        self.B = matrizB
        
    
    def obtain_f(self):        
        f = np.zeros((1, self.x_size))       
      #  f[0, self.n - 1:(self.n+self.m) - 1] = -1
      
        f[0, :self.n - 1] = 10         # Penaliza tensiones elevadas (Ckk)
        #f[0, self.n - 1:(self.n + self.m) - 1] = -0.1  # Sigue minimizando Ckt suavemente
       
        # No optimizamos directamente los flujos
        f[0, self.n - 1:(self.n + self.m) - 1] = 0.0
        self.f = f
    
    def compute_voltages(self):                  
        self.nodes[0].U = complex(1, 0)
        for line in self.lines:
            A = np.array([[np.real(line.nodes[0].U), np.imag(line.nodes[0].U)], 
                          [-np.imag(line.nodes[0].U), np.real(line.nodes[0].U)]], dtype = np.float64)
            b = np.array([line.Ckt, line.Skt], dtype = np.float64)
            x = np.linalg.solve(A, b)
            line.nodes[1].U = complex(x[0], x[1])           
        return [node.U for node in self.nodes]  
          
    
class node:
    def __init__(self, ref, slack):
        self.ref = ref   
        self.slack = slack        
        self.lines = list()
        self.pros = []
        self.Ckk = None
        self.Ctt = None
        self.index = None
        self.U = None
        
class line:
    def __init__(self, ref, From, To, R, X, nodes_list):
        self.ref = ref     
        self.Z = complex(R, X)
        self.G, self.B = np.real(1/self.Z), -np.imag(1/self.Z)
        self.Y = 1/self.Z
        self.nodes = [next((item for item in nodes_list if item.ref == From), None), 
                      next((item for item in nodes_list if item.ref == To), None)]   
        self.nodes[0].lines.append(self)
        self.nodes[1].lines.append(self)
        self.Ckt = None
        self.Skt = None
        self.index = []  
        
    def ineq(self, X):
        if self.nodes[0].slack == True:
            Ckk = 1
        else:
            Ckk = X[self.nodes[0].index]        
        Ctt = X[self.nodes[1].index]
        self.Ckt = X[self.index[0]]
        self.Skt = X[self.index[1]]
        # ineq = self.Ckt**2 + self.Skt**2 - Ckk * Ctt
        # return ineq
        # Calcular la inecuación con protección
        try:
           ineq = self.Ckt**2 + self.Skt**2 - Ckk * Ctt
           if not np.isfinite(ineq):
               return 1e10  # penaliza si es NaN o inf
           return ineq
        except:
           return 1e10  # penaliza si hay error matemático
     
    def ineq_jac(self, X):
        jac = np.zeros(len(X))  # Inicializar con ceros
        
        # Obtener los índices
        idx_Ckk = self.nodes[0].index if not self.nodes[0].slack else None
        idx_Ctt = self.nodes[1].index
        idx_Ckt = self.index[0]
        idx_Skt = self.index[1]
        
        # Obtener los valores
        Ckk = 1 if self.nodes[0].slack else X[idx_Ckk]
        Ctt = X[idx_Ctt]
        Ckt = X[idx_Ckt]
        Skt = X[idx_Skt]
        
        # Protección contra valores extremos
        try:
            if not all(np.isfinite([Ckk, Ctt, Ckt, Skt])):
                return jac  # Devuelve 0 si hay valores no finitos
        
            # Calcular derivadas parciales
            jac[idx_Ckt] = 2 * Ckt
            jac[idx_Skt] = 2 * Skt
            if idx_Ckk is not None:
                jac[idx_Ckk] = -Ctt
            jac[idx_Ctt] = -Ckk
        
        except:
            pass  # Si algo falla, devolvemos el jacobiano con ceros
        
        return jac   
     
    def current(self):
        self.I = (self.nodes[0].U - self.nodes[1].U) / self.Z   
        return self.I
            
class prosumer:
    def __init__(self, ref, node_id, P, Q, nodes_list):
        self.ref = ref
        self.P = P
        self.Q = Q    
        self.S = complex(self.P, self.Q)
        self.node = next((item for item in nodes_list if item.ref == node_id), None)
        self.node.pros.append(self)
        
    def current(self):
        self.I = np.conj(self.S/(self.node.U))
        return self.I