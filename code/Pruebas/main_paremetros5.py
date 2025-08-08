# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:22:12 2025

@author: aberzal
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import lib_itterative_v5
import time

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

# Generar combinaciones de parámetros
alphas = [1e-9]
rhos = [40]
param_grid = list(itertools.product(alphas, rhos))

hist_eq_all = []
hist_ineq_all = []
hist_dual_all = []
registro_estados = []

print("\n🔍 Exploración fina de parámetros...\n")

for i, (alpha, rho) in enumerate(param_grid):
    etiqueta = f"α={alpha:.0e}, ρ={rho:.0e}"
    print(f"[#{i+1:02d}] {etiqueta}...", end=' ')
    grid_obj = lib_itterative_v5.grid(Nodes, Lines, Pros)
    grid_obj.initialize()

    try:
        converged, x, k, estado, (res_eq, res_ineq, res_dual), hist_eq, hist_ineq, hist_dual = grid_obj.solve_iterative(
            alpha=alpha,
            rho=rho,
            max_iter=150000,
            tol=1e-5,
            verbose=False,
            ventana_estancamiento=150000
        )

        if not any(np.isnan(hist_eq)) and not any(np.isnan(hist_ineq)) and not any(np.isnan(hist_dual)):
            hist_eq_all.append((etiqueta, hist_eq))
            hist_ineq_all.append((etiqueta, hist_ineq))
            hist_dual_all.append((etiqueta, hist_dual))
            registro_estados.append({'etiqueta': etiqueta, 'estado': estado, 'iteraciones': k,
                                     'res_eq': res_eq, 'res_ineq': res_ineq, 'res_dual': res_dual})
            print(f"{estado.upper()}")
        else:
            print("⚠️ NaN en residuos. Ignorada.")

    except Exception as e:
        print(f"❌ Error: {e}")

# ---------------------------------------------
# GRAFICADO FINAL
# ---------------------------------------------
def graficar(historial, titulo, ylabel):
    plt.figure(figsize=(10, 6))
    for etiqueta, hist in historial:
        plt.plot(hist, label=etiqueta, linewidth=0.8)
    plt.title(titulo)
    plt.xlabel("Iteración")
    plt.ylabel(ylabel)
    #plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.show()

graficar(hist_eq_all, "Residuo de igualdad (Ax = B)", "||Ax - B||∞")
graficar(hist_ineq_all, "Residuo de desigualdad (g(x) + s)", "||g(x) + s||∞")
graficar(hist_dual_all, "Residuo dual (gradiente)", "||∇ₓL||∞")

# ---------------------------------------------
# EXPORTAR RESIDUOS Y ESTADOS A CSV
# ---------------------------------------------
def construir_dataframe(historial, tipo):
    df = pd.DataFrame()
    for etiqueta, hist in historial:
        df_temp = pd.DataFrame({
            'iteracion': np.arange(len(hist)),
            'residuo': hist,
            'tipo': tipo,
            'etiqueta': etiqueta
        })
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

df_eq = construir_dataframe(hist_eq_all, 'igualdad')
df_ineq = construir_dataframe(hist_ineq_all, 'desigualdad')
df_dual = construir_dataframe(hist_dual_all, 'dual')

df_total = pd.concat([df_eq, df_ineq, df_dual], ignore_index=True)
df_total.to_csv("residuos_iterativos_fino.csv", index=False)

df_estados = pd.DataFrame(registro_estados)
df_estados.to_csv("resumen_iterativo_fino.csv", index=False)

print("\n📁 Archivos 'residuos_iterativos_fino.csv' y 'resumen_iterativo_fino.csv' generados.")