# Importing required libraries
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint


class grid:
    def __init__(self, nodes, lines, pros):
        self.nodes = self.add_nodes(nodes)                                      
        self.lines = self.add_lines(lines, self.nodes)  
        self.pros = self.add_pros(pros, self.nodes)  
        self.n = len(self.nodes)
        self.m = len(self.lines)
        self.x_size = self.n + 2*self.m
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
        for idx, node in enumerate(self.nodes):
            node.index = idx
        for idx, line in enumerate(self.lines):
            line.index = [self.n + idx, self.n + self.m + idx]      
        self.x = np.zeros(self.x_size)
        self.x[:self.n + self.m] = 1
        for line in self.lines:
            line.mask = np.array([False for _ in range(self.x_size)])
            line.mask[line.index[0]] = True
            line.mask[line.index[1]] = True
            line.mask[line.nodes[0].index] = True
            line.mask[line.nodes[1].index] = True
        
    def compute_A(self):        
        self.A = np.zeros((2*self.n-2+1, 
                           self.n+2*self.m), 
                          dtype=float)        
        self.A[0, next((node.index for i, node in enumerate(self.nodes) if node.slack), None)] = 1
        for idx, node in enumerate(self.nodes[1:]):
            self.A[1 + 2*idx, node.index] = np.sum([line.G for line in node.lines])
            self.A[2 + 2*idx, node.index] = np.sum([line.B for line in node.lines])            
            
            for idj, line in enumerate(node.lines):
                if node == line.nodes[0]:
                    self.A[1 + 2*idx, line.index[0]] = -line.G
                    self.A[1 + 2*idx, line.index[1]] = -line.B
                    self.A[2 + 2*idx, line.index[0]] = -line.B
                    self.A[2 + 2*idx, line.index[1]] =  line.G
                else:
                    self.A[1 + 2*idx, line.index[0]] = -line.G
                    self.A[1 + 2*idx, line.index[1]] =  line.B
                    self.A[2 + 2*idx, line.index[0]] = -line.B
                    self.A[2 + 2*idx, line.index[1]] = -line.G    
        for jdx in range(self.n):
            for idx in range(self.A.shape[0]):
                self.A[idx, jdx] = self.A[idx, jdx]*np.sqrt(2)
      
    def compute_B(self):        
        self.B = np.zeros(2*self.n-2+1, dtype=float) 
        self.B[0] = 1
        for idx, node in enumerate(self.nodes[1:]):           
            for x in node.pros:
                self.B[1 + 2*idx] += x.P
                self.B[2 + 2*idx] += x.Q           
    
    def compute_f(self):        
        self.f = np.zeros(self.x_size)       
        self.f[self.n:self.n + self.m] = -1     
    
    def compute_Q(self):
        self.Q = list()
        for line in self.lines:
            self.Q.append(line.compute_Q(self.x_size))       
            
    def compute_G(self):
        self.G = np.zeros((0, self.x_size))
        for line in self.lines:
            self.G = np.block([[self.G],
                               [line.compute_G(self.x_size)]])    
        self.G = self.G[~np.all(self.G == 0, axis=1)] 
        self.G = np.unique(self.G, axis=0)
        return self.G
            
    def ineq(self, x):        
        rest = []
        for q in self.Q:
            rest.append(x @ q @ x)          
        return rest   
    
    def project_Q(self, x):
        if x @ self.Q[0] @ x <= 0 and x[0] >= 0 and x[1] >= 0:
            return x
        elif x[0] >= 0 and x[1] < 0:
            x[1] = 0
            x[2], x[3] = 0, 0        
            return x
        elif x[1] >= 0 and x[0] < 0:
            x[0] = 0
            x[2], x[3] = 0, 0
            return x
        else:
            A = np.zeros((2, 4))
            A[0, 0] = 1
            A[1, 1] = 1
            lc = LinearConstraint(A, 0, np.inf)
            nlc = NonlinearConstraint(lambda y: y @ self.Q[0] @ y, -np.inf, 0)
            fo = lambda y: np.linalg.norm(x - y)
            sol = minimize(fo, x, constraints=(lc, nlc)) 
            x = sol.x
            return x
                
                
    
    def check(self):
        residuals = []
        for node in self.nodes[1:]:
            res = node.I 
            for line in node.lines:
                if line.nodes[0] == node:
                    res -= line.I
                else:
                    res += line.I
            residuals.append(res)
        return residuals             
    
    def compute_mags(self):  
        for node in self.nodes:
            if node.slack:
                node.U = 1
        for line in self.lines:
            A = np.array([[np.real(line.nodes[0].U), np.imag(line.nodes[0].U)], 
                          [-np.imag(line.nodes[0].U), np.real(line.nodes[0].U)]], dtype = np.float64)
            b = np.array([line.Ckt, line.Skt], dtype = np.float64)
            x = np.linalg.solve(A, b)
            line.nodes[1].U = complex(x[0], x[1])   
            
        for node in self.nodes:
            node.compute_I()
        for line in self.lines:
            line.compute_I()     
                      
    
    def solve_pf(self):        
        self.compute_A()
        self.compute_B()
        self.compute_Q()
        self.compute_G()
        self.compute_f()    
        
        lc = LinearConstraint(self.A, self.B, self.B)
        nlc = NonlinearConstraint(self.ineq, 0, np.inf)
        fo = lambda x: self.f.dot(x)
        sol = minimize(fo, self.x, constraints=(lc, nlc))    
        print(sol.message)        
        self.x = sol.x
        
        for node in self.nodes:
            node.Ckk = sol.x[node.index]    
        for line in self.lines:
            line.Ckt = sol.x[line.index[0]]
            line.Skt = sol.x[line.index[1]]
            
        self.compute_mags()
        
        return sol   
      
    def solve_iterative(self, rho, max_iter, tol):           
        self.compute_A()
        self.compute_B()
        self.compute_Q()
        self.compute_f()  
                
        # Inicializamos varibales 
        self.z = [self.x.copy()*line.mask for line in self.lines]
        self.y = np.zeros(self.A.shape[0])
        self.u = [np.zeros(self.x_size) for _ in self.lines]
            
        d = np.zeros(self.x_size)
        for line in self.lines:
            d[line.index[0]] += 1 
            d[line.index[1]] += 1  
            d[line.nodes[0].index] += 1   
            d[line.nodes[1].index] += 1   
        self.D = np.diag(d)      
        
        it = 0
        res_out = [100] + [100 for z_i in self.z] + [100 for z_i in self.z]  
        while any(x > tol for x in res_out) and it < max_iter:
            
            # Updating x
            A_ = rho*(self.A.T @ self.A + self.D)
            B_ = - self.f + rho*self.A.T @ (self.B - (1/rho)*self.y) + rho*np.sum(np.array([zu_i[0] - (1/rho)*zu_i[1] for zu_i in zip(self.z, self.u)]), axis=0)
            self.x = np.linalg.solve(A_, B_)
            
            # Updating z
            z_old = [z_i.copy() for z_i in self.z]            
            self.z = [self.x + (1/rho)*u_i for u_i in self.u]
            self.z = [self.z[idx]*self.lines[idx].mask for idx in range(len(self.lines))]
            for idx in range(len(self.lines)):
                self.z[idx] = self.lines[idx].project_cone(self.z[idx])
                
            # Updating dual variables
            self.y += rho*(self.A @ self.x - self.B)
            for idx in range(len(self.u)):
                self.u[idx] = self.u[idx] + rho*(self.x*self.lines[idx].mask - self.z[idx]*self.lines[idx].mask)
                                    
            # Update counter and residuals
            it +=1
            res_out = [np.linalg.norm(self.A @ self.x - self.B)] + [np.linalg.norm(self.x*self.lines[idx].mask - self.z[idx]) for idx in range(len(self.lines))] + [rho*np.linalg.norm(z_i[0] - z_i[1]) for z_i in zip(z_old, self.z)]  
            
            print(f'Iter {it}: \n\tAx=b:\t\t\t{res_out[0]} \n\tx-z_i:\t\t\t{res_out[1:len(self.lines)+1]} \n\tz(k+1)-z(k):\t{res_out[len(self.lines)+1:]} ')
        
        print(f'\n Iteration {it} \n')
        return self.x
           
    
    
class node:
    def __init__(self, ref, slack):
        self.ref = ref   
        self.slack = slack        
        self.lines = list()
        self.pros = []
        self.Ckk = None
        self.index = None
        self.U = None
        
    def compute_I(self):
        self.I = np.sum([p.compute_I() for p in self.pros])
            
        
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
    
    def compute_Q(self, size_x):
        self.Q = np.zeros((size_x, size_x))
        self.Q[self.index[0], self.index[0]], self.Q[self.index[1], self.index[1]] = -1, -1 
        self.Q[self.nodes[0].index, self.nodes[1].index], self.Q[self.nodes[1].index, self.nodes[0].index] = 1, 1
        return self.Q     
    
    def compute_G(self, size_x):
        self.Gpd = np.zeros((4, size_x))
        self.Gpd[0, self.index[0]] = -1
        self.Gpd[1, self.index[1]] = -1
        self.Gpd[2, self.nodes[0].index] = -1/np.sqrt(2)
        self.Gpd[3, self.nodes[1].index] = -1/np.sqrt(2)
        return self.Gpd
    
    def project_cone(self, s):
        w = [s[self.index[0]], s[self.index[1]]]
        u = s[self.nodes[0].index]
        v = s[self.nodes[1].index]
        if u >= 0 and v>= 0 and 2*u*v >= np.linalg.norm(w)**2:
            return s
        elif u <= 0 and v <= 0:
            s[self.index[0]], s[self.index[1]] = 0, 0
            s[self.nodes[0].index] = 0
            s[self.nodes[1].index] = 0
            return s
        elif u <= 0 and v > 0:
            s[self.index[0]], s[self.index[1]] = 0, 0
            s[self.nodes[0].index] = 0
            return s
        elif u > 0 and v <= 0:
            s[self.index[0]], s[self.index[1]] = 0, 0
            s[self.nodes[1].index] = 0
            return s
        else:            
            alpha = np.linalg.norm(w)**2
            a = 2*u*v - alpha
            b = -2*(u**2+ v**2 + alpha)
            c = 2*u*v - alpha
            mu = np.roots([a, b, c])
            idx = np.argmin(np.abs(mu))
            mu = mu[idx]
            up = (u - mu*v)/(1 - mu**2)
            vp = (v - mu*u)/(1 - mu**2)
            wp = [(1/(1 - mu))*w[0], (1/(1 - mu))*w[1]]            
            s[self.nodes[0].index] = up
            s[self.nodes[1].index] = vp
            s[self.index[0]] = wp[0]
            s[self.index[1]] = wp[1]
            return s
            
         
    def compute_I(self):
        self.I = (self.nodes[0].U - self.nodes[1].U) / self.Z   
            
class prosumer:
    def __init__(self, ref, node_id, P, Q, nodes_list):
        self.ref = ref
        self.P = P
        self.Q = Q    
        self.S = complex(self.P, self.Q)
        self.node = next((item for item in nodes_list if item.ref == node_id), None)
        self.node.pros.append(self)
        
    def compute_I(self):
        self.I = np.conj(self.S/(self.node.U))
        return self.I