# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 10:07:32 2025

@author: aberzal
"""

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
        #alpha = 0.00001            # paso inicial
        tol = 1e-6
        max_inner_iter = 1000
        obj_scale = 2.0  # peso extra a la función objetivo
        rho_eq_factor = 1.0  # más peso a las restricciones de igualdad
        Vmax2 = 1.21  # 🔹 límite físico de Ckk y Ctt (1.1^2 pu)
        
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
                #s = np.maximum(0, -gx - mu / rho) #modifico la funcion S por una equivalente por si es error de python
                s = np.where(gx > 0, gx, 0)  #solo si hay violacion
                
                # Paso 5: gradiente Lagrangiano aumentado
                grad_f = obj_scale * self.f.flatten()
                jac_g = self.ineq_jac(x)
               
                penalty = mu + rho * (gx + s)
                
                grad_L = grad_f + rho_eq_factor * (self.A.T @ lam) + jac_g.T @ penalty

                
                # Paso 6: Errores antes de decidir alpha
                error_grad = np.linalg.norm(grad_L)
                error_eq = np.linalg.norm(self.A @ x - self.B)

                # Alpha adaptativo con límites
                grad_norm = max(error_grad, 1e-8)
                alpha = 5e-3 / grad_norm
                alpha = np.clip(alpha, 1e-4, 5e-3)  # Evitar pasos muy pequeños
                
                if error_eq < 1e-2 and error_grad < 1e-2:
                    alpha *= 0.5
                
               

                # Actualización primal
                x = x - alpha * grad_L
                x = np.clip(x, -10, 10)

                # Proyección física y cónica
                for line in self.lines:
                    i_Ckk = line.nodes[0].index if not line.nodes[0].slack else None
                    i_Ctt = line.nodes[1].index
                    i_Ckt, i_Skt = line.index

                    # Clip físico de Ckk y Ctt
                    if i_Ckk is not None:
                        x[i_Ckk] = np.clip(x[i_Ckk], 1e-6, Vmax2)
                    x[i_Ctt] = np.clip(x[i_Ctt], 1e-6, Vmax2)

                   # Recuperar valores ya corregidos
                    Ckk = 1 if line.nodes[0].slack else x[i_Ckk]
                    Ctt = x[i_Ctt]
                    Ckt = x[i_Ckt]
                    Skt = x[i_Skt]

                    # Límite físico en Ckt y Skt
                    max_flow = np.sqrt(Ckk * Ctt)
                    flow_norm = np.sqrt(Ckt**2 + Skt**2)
                    if flow_norm > max_flow:
                        scale = max_flow / (flow_norm + 1e-12)
                        Ckt *= scale
                        Skt *= scale
    
                    # Proyección cónica si hay violación
                    g_val = Ckt**2 + Skt**2 - Ckk * Ctt
                    if g_val > 0:
                        factor = np.sqrt((Ckk * Ctt) / (Ckt**2 + Skt**2 + 1e-12))
                        factor = 0.5 * (1 + factor)
                        Ckt *= factor
                        Skt *= factor
    
                    # Guardar valores corregidos
                    x[i_Ckt] = Ckt
                    x[i_Skt] = Skt


                
                # 🔹 Actualizar duales al final
                lam = lam + rho_eq_factor * rho * (self.A @ x - self.B)
                mu = np.maximum(0, mu + rho * (gx + s))
                
                
                
                # 🧪 Depuración cada 100 iteraciones
                if k % 100 == 0:
                    print(f"Iter {k:4d} | grad={error_grad:.3e} | eq={error_eq:.3e}")
                
                      # Condición de parada
                if error_grad < tol and error_eq < tol and np.all(gx <= 0):
                     print(f"✅ Convergencia alcanzada en {k} pasos internos.")
                     break
       
       
               
            # Guardar la solución de x y actualizar Ckk para ver tensiones
            self.x = x
            for index, node in enumerate(self.nodes[1:]):
                node.Ckk = x[index]
            self.compute_voltages()
            
            # Seguimiento: tensiones y flujos al final de outer loop
            voltajes = [abs(n.U) for n in self.nodes]
            flujos = self.compute_line_currents()
           
            print(f"Tensiones (pu): {[round(v,4) for v in voltajes]}")
            print(f"Flujos (A): {[round(abs(I),4) for I in flujos]}") 
           
            # Condición de parada externa
            if np.all(gx <= 0) and error_eq < tol and error_grad < tol:
                print("✅ Restricciones cónicas y balance nodal satisfechos.")
                break
            else:
                print("⚠️ Aún hay violaciones. Aumentando rho...")
                rho *= 2  # aumento suave


        ##################################################
        
        return x
        
    
    
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
      
     #   f[0, :self.n - 1] = 10         # Penaliza tensiones elevadas (Ckk)
        f[0, self.n - 1:(self.n + self.m) - 1] = -0.1  # Sigue minimizando Ckt suavemente
       
        # No optimizamos directamente los flujos
      #  f[0, self.n - 1:(self.n + self.m) - 1] = 0.0
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
        ineq = self.Ckt**2 + self.Skt**2 - Ckk * Ctt
        return ineq
        # Calcular la inecuación con protección
        #try:
          # ineq = self.Ckt**2 + self.Skt**2 - Ckk * Ctt
          # if not np.isfinite(ineq):
            #   return 1e10  # penaliza si es NaN o inf
          # return ineq
      #  except:
         #  return 1e10  # penaliza si hay error matemático
     
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