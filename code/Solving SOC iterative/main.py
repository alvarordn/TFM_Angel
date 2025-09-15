#%% Scipy
import lib
import numpy as np

# Datos de partida
Sbase = 1e6
Ubase = 20e3
Zbase = (Ubase**2)/Sbase

Nodes = [{'id': 0, 'slack': True },
          {'id': 1, 'slack': False},
          {'id': 2, 'slack': False},
          {'id': 3, 'slack': False}]

Lines = [{'id': 0, 'From': 0, 'To': 1, 'R': 0.161*4/Zbase, 'X': 0.109*4/Zbase},
          {'id': 1, 'From': 1, 'To': 2, 'R': 0.161*2/Zbase, 'X': 0.109*2/Zbase},
          {'id': 2, 'From': 2, 'To': 3, 'R': 0.161*5/Zbase, 'X': 0.109*5/Zbase}]

Pros = [{'id': 0, 'Node': 1, 'P': -2e6/Sbase, 'Q': -1.5e6/Sbase},
        {'id': 1, 'Node': 2, 'P': -1.6e6/Sbase, 'Q': -1.2e6/Sbase},
        {'id': 2, 'Node': 3, 'P': -6.4e6/Sbase, 'Q': -2.4e6/Sbase}]


#%% ADMM

print('\n\n\n')
print('------------- ITERATIVE -------------')
net = lib.grid(Nodes, Lines, Pros)
net.solve_iterative(rho=1000,
                    max_iter = 5000, 
                    tol = 1e-3)
print('Linear constraints: ')
res_eq = net.A @ net.x - net.B
for item in res_eq: 
    print(f'\t {item}')
    
print('\n Conic constraints: ')
res_ineq = net.ineq(net.x)
for item in res_ineq: 
    print(f'\t {item}')
    
print('')
print(net.x)
x_iterative = net.x

#%% Resolvemos con scipy y comprobamos
print('\n\n\n')
print('------------- SCIPY -------------')
net = lib.grid(Nodes, Lines, Pros)
net.solve_pf()

print(net.x)
x_scipy = net.x

#%% Comparamos
res = x_iterative - x_scipy
print('\n')
print(f'Differences: {np.linalg.norm(res)}')






















