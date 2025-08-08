# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:18:47 2025

@author: aberzal
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import lib
import lib_itterative_v5

# BASES
Sbase = 1e6
Ubase = 20e3
Zbase = (Ubase**2)/Sbase

# NODOS
Nodes = [{'id': 0, 'slack': True},
         {'id': 1, 'slack': False},
         {'id': 2, 'slack': False},
         {'id': 3, 'slack': False}]

# LÍNEAS
Lines = [{'id': 0, 'From': 0, 'To': 1, 'R': 0.161*4/Zbase, 'X': 0.109*4/Zbase},
         {'id': 1, 'From': 1, 'To': 2, 'R': 0.161*2/Zbase, 'X': 0.109*2/Zbase},
         {'id': 2, 'From': 2, 'To': 3, 'R': 0.161*5/Zbase, 'X': 0.109*5/Zbase}]

# CARGAS
Pros = [{'id': 0, 'Node': 1, 'P': -2e6/Sbase, 'Q': -1.5e6/Sbase},
        {'id': 1, 'Node': 2, 'P': -1.6e6/Sbase, 'Q': -1.2e6/Sbase},
        {'id': 2, 'Node': 3, 'P': -6.4e6/Sbase, 'Q': -2.4e6/Sbase}]

# REFERENCIA SCIPY
net_scipy = lib.grid(Nodes, Lines, Pros)
net_scipy.solve_pf()
U_scipy = net_scipy.compute_voltages()

# ---------- PRUEBAS MULTIPARAMÉTRICAS ----------
alphas = [1e-8, 2e-8, 5e-8]
rhos = [1e-2, 5e-3, 1e-3]
combinaciones = list(itertools.product(alphas, rhos))

print("\n🧪 Lanzando pruebas sobre combinaciones de alpha y rho...\n")
for i, (alpha, rho) in enumerate(combinaciones):
    print(f"🔹 Prueba #{i+1}: alpha = {alpha:.0e}, rho = {rho:.0e}")

    net_iter = lib_itterative_v5.grid(Nodes, Lines, Pros)
    converged, iters, (res_eq, res_ineq, res_dual) = net_iter.solve_iterative(
        alpha=alpha,
        rho=rho,
        max_iter=10000,
        tol=1e-6,
        verbose=False
    )
    U_iter = net_iter.compute_voltages()

    print("Convergencia:", "✅" if converged else "❌", f"| Iteraciones: {iters} | Residuos: eq={res_eq:.1e}, ineq={res_ineq:.1e}, dual={res_dual:.1e}")

    print("\nComparación de tensiones por nodo:")
    print("Nodo |  Iterativo (pu) |  SciPy (pu)  |  Δ Abs     |  Δ Rel (%)")
    for j, (u1, u2) in enumerate(zip(U_iter, U_scipy)):
        abs_diff = abs(abs(u1) - abs(u2))
        rel_diff = 100 * abs_diff / abs(u2) if abs(u2) > 0 else 0
        print(f"{j:^4} |  {abs(u1):^15.6f} | {abs(u2):^12.6f} | {abs_diff:^9.2e} | {rel_diff:^9.2f}")

    print("\n¿Se cumplen las restricciones cónicas?")
    for line in net_iter.lines:
        g = line.ineq(net_iter.x)
        print(f"Línea {line.ref}: g(x) = {g:.3e} → {'OK' if g <= 0 else '❌'}")

    print("-" * 60)

# ---------- OPCIONAL: GRÁFICA ÚLTIMA SIMULACIÓN ----------
labels = [f"Nodo {i}" for i in range(len(U_iter))]
v_iter = [abs(u) for u in U_iter]
v_scipy = [abs(u) for u in U_scipy]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, v_iter, width, label='Iterativo')
ax.bar(x + width/2, v_scipy, width, label='Scipy (minimize)')

ax.set_ylabel('Tensión (pu)')
ax.set_title(f'Comparación tensiones - Última prueba (α={alpha:.0e}, ρ={rho:.0e})')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()