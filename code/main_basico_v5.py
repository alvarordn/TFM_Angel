import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lib

# Datos de partida
Sbase = 1e6
Ubase = 20e3
Zbase = (Ubase**2)/Sbase

Nodes = [{'id': 0, 'slack': True},
         {'id': 1, 'slack': False},
         {'id': 2, 'slack': False},
         {'id': 3, 'slack': False}]

Lines = [{'id': 0, 'From': 0, 'To': 1, 'R': 0.161*4/Zbase, 'X': 0.109*4/Zbase},
         {'id': 1, 'From': 1, 'To': 2, 'R': 0.161*2/Zbase, 'X': 0.109*2/Zbase},
         {'id': 2, 'From': 2, 'To': 3, 'R': 0.161*5/Zbase, 'X': 0.109*5/Zbase}]

Pros = [{'id': 0, 'Node': 1, 'P': -2e6/Sbase, 'Q': -1.5e6/Sbase},
        {'id': 1, 'Node': 2, 'P': -1.6e6/Sbase, 'Q': -1.2e6/Sbase},
        {'id': 2, 'Node': 3, 'P': -6.4e6/Sbase, 'Q': -2.4e6/Sbase}]


# Resolvemos con scipy y comprobamos
net_scipy = lib.grid(Nodes, Lines, Pros)
net_scipy.solve_pf()
U_scipy = net_scipy.compute_voltages()
v_ref = np.array([abs(u) for u in U_scipy])

print('------------- SCIPY -------------')
print(net_scipy.x)
print(net_scipy.check())
print(net_scipy.A @ net_scipy.x - net_scipy.B)
print(net_scipy.ineq(net_scipy.x))
# print('Tolerancia a considerar...')
# print(np.linalg.norm(net_scipy.A @ net_scipy.x - net_scipy.B, np.inf))


# Resolvemos con algorimto iterativo
print('------------- ITERATIVE -------------')
net = lib.grid(Nodes, Lines, Pros)
# net.x = net_scipy.x
net.solve_iterative(
    alpha=1e-8,
    rho=0.1,
    max_iter=20,
    tol=1e-5,
    verbose=True
)

    
U_iterative = net.compute_voltages()

print(net.x)
print(net.check())
# print(net.A @ net.x - net.B)
# print(net.ineq(net.x))
