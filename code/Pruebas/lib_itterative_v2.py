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
        
            ##################################################
        """
        Algoritmo iterativo primal descent + dual ascent
        con variables de holgura y Lagrangiano aumentado (versión base).
        Paso 1: Inicialización de variables y parámetros.
        """
    
        # --- Paso 1: Preparar matrices y vectores del problema ---
        self.obtain_A()   # Matriz de igualdades Ax = B
        self.obtain_B()   # Vector de igualdades
        self.obtain_f()   # Vector de costes f
    
        # --- Inicialización de variables ---
        x = self.x.copy()                                  # Variables primales
        s = np.zeros(len(self.ineq(x)))                    # Variables de holgura (para desigualdades)
        lambda_eq = np.zeros(self.A.shape[0])               # Multiplicadores para igualdades
        mu_ineq = np.zeros(len(self.ineq(x)))               # Multiplicadores para desigualdades
    
        # --- Parámetros de control ---
        alpha = 1e-9   # Paso primal y dual
        rho = 20.0      # Penalización del Lagrangiano aumentado
        tol = 1e-7     # Tolerancia de convergencia máxima
        max_iter = 10000
    
        # Guardamos histórico opcional (para análisis posterior)
        hist_res_primal = []
        hist_res_dual = []
    
        # --- Aquí empezaría el bucle iterativo ---
        for k in range(max_iter):
            
            # --- Paso 2: Actualización de variables de holgura ---
            g_x = np.array(self.ineq(x))  # Evaluar desigualdades g(x)
            s = -mu_ineq / rho - g_x      # Solución analítica de ∂L/∂s = 0
            s = np.maximum(s, 0)          # Imponer condición s >= 0
           
            # --- Paso 3: Actualización de variables primales x ---
            # Gradiente del Lagrangiano aumentado respecto a x
            grad_x = (self.f.flatten() +
                      self.A.T @ lambda_eq +
                      self.ineq_jac(x).T @ mu_ineq +
                      rho * self.A.T @ (self.A @ x - self.B) +
                      rho * self.ineq_jac(x).T @ (g_x + s))
    
            # Paso de descenso primal
            x = x - alpha * grad_x
            
            # --- Paso 4: Actualización de variables duales ---
            lambda_eq = lambda_eq + alpha * (self.A @ x - self.B)
            mu_ineq = mu_ineq + alpha * (g_x + s)
            mu_ineq = np.maximum(mu_ineq, 0)  # Proyección para mantener μ >= 0
               
           # --- Paso 5: Criterio de parada ---
            res_eq = np.linalg.norm(self.A @ x - self.B, np.inf)
            res_ineq = np.linalg.norm(g_x + s, np.inf)
            res_dual = np.linalg.norm(grad_x, np.inf)
    
            hist_res_primal.append(max(res_eq, res_ineq))
            hist_res_dual.append(res_dual)

    # --- Trazas de depuración ---
            if k % 100 == 0:
                max_x = np.max(np.abs(x))
                max_s = np.max(np.abs(s))
                max_mu = np.max(np.abs(mu_ineq))
                max_grad = np.max(np.abs(grad_x))
                print(f"Iter {k:4d} | max|x|={max_x:.3e} | max|s|={max_s:.3e} | max|mu|={max_mu:.3e} | max_grad={max_grad:.3e}")



    
            if res_eq < tol and res_ineq < tol and res_dual < tol:
                print(f"Convergencia alcanzada en iteración {k}")
                break
        else:
            print("No se alcanzó convergencia en el número máximo de iteraciones")
    
        # Guardar solución final
        self.x = x
                
            
           ################################## 
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
        f[0, self.n - 1:(self.n+self.m) - 1] = -1
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
        
        # Calcular derivadas parciales
        jac[idx_Ckt] = 2 * Ckt
        jac[idx_Skt] = 2 * Skt
        if idx_Ckk is not None:
            jac[idx_Ckk] = -Ctt
        jac[idx_Ctt] = -Ckk
        
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