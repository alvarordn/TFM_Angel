# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 19:44:30 2025

@author: aberzal
"""

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import itertools
import lib_itterative_v4
import time
import pandas as pd
import os


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

# CARGAS (Prosumidores)
Pros = [{'id': 0, 'Node': 1, 'P': -2e6/Sbase, 'Q': -1.5e6/Sbase},
        {'id': 1, 'Node': 2, 'P': -1.6e6/Sbase, 'Q': -1.2e6/Sbase},
        {'id': 2, 'Node': 3, 'P': -6.4e6/Sbase, 'Q': -2.4e6/Sbase}]

# ------------------------------
# ⚙️ CONFIGURACIÓN DE PARÁMETROS
# ------------------------------
alpha_x = 1e-8
alpha_lambda = 1e-7
alpha_mu = 1.2e-6
rhos = [1e-2, 2e-2, 4e-2]
tol = 1e-4

# ------------------------------
# 🧪 PRUEBAS COMBINADAS adapt_rho + adapt_alpha
# ------------------------------
resultados = []

for rho_init in rhos:
    for adapt_rho, adapt_alpha in itertools.product([False, True], repeat=2):
        modo = f"{'r' if adapt_rho else '-'}{'a' if adapt_alpha else '-'}"

        grid = lib_itterative_v4.grid(Nodes, Lines, Pros)
        grid.initialize()

        converged, iters, (res_eq, res_ineq, res_dual), \
        _, _, _, _, _, _, rho_hist, alpha_hist = grid.solve_iterative(
            alpha_x=alpha_x,
            alpha_lambda=alpha_lambda,
            alpha_mu=alpha_mu,
            rho=rho_init,
            max_iter=30000,
            tol=tol,
            verbose=False,
            adapt_rho=adapt_rho,
            adapt_alpha=adapt_alpha
        )

        resultados.append({
            'modo': modo,
            'rho_inicial': rho_init,
            'res_eq': res_eq,
            'res_ineq': res_ineq,
            'res_dual': res_dual,
            'iters': iters,
            'convergido': converged
        })

        print(f"[{modo}] ρ₀={rho_init:.0e} → iters={iters}, res_eq={res_eq:.1e}, res_ineq={res_ineq:.1e}, res_dual={res_dual:.1e}, {'✅' if converged else '❌'}")

# ------------------------------
# 📋 TABLA COMPARATIVA FINAL
# ------------------------------
df = pd.DataFrame(resultados)
print("\n📊 Comparativa combinada adapt_rho vs adapt_alpha:\n")
print(df.sort_values(by='res_dual').to_string(index=False))

# ------------------------------
# 📈 EVOLUCIÓN DE αx POR ITERACIÓN (modo adaptativo)
# ------------------------------
plt.figure(figsize=(10, 6))

for rho_init in rhos:
    grid = lib_itterative_v4.grid(Nodes, Lines, Pros)
    grid.initialize()

    result = grid.solve_iterative(
        alpha_x=alpha_x,
        alpha_lambda=alpha_lambda,
        alpha_mu=alpha_mu,
        rho=rho_init,
        max_iter=30000,
        tol=tol,
        verbose=False,
        adapt_rho=False,
        adapt_alpha=True
    )

    alpha_hist = result[-1] if len(result) == 12 else []
    if alpha_hist:
        plt.plot(alpha_hist, label=f"ρ₀ = {rho_init:.0e}")

plt.xlabel("Iteración")
plt.ylabel("Valor de αₓ")
plt.yscale("log")
plt.title("Evolución adaptativa de αₓ por iteración")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()