# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 19:19:21 2025

@author: aberzal
"""

"""
Adaptado para lib_itterative_v5.py con alpha=1e-8 y rho=40
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lib
import lib_itterative_v5
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# ---------- BASES Y DATOS INMUTABLES ----------

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

# ---------- RANGOS DE ALPHA Y RHO ----------

alphas = np.geomspace(1e-10, 1e-9, num=2)
rhos = np.geomspace(10, 284, num=10)
max_iter = 20000
tol = 1e-5
combinaciones = [(a, r) for a in alphas for r in rhos]

# ---------- REFERENCIA SCIPY ----------

net_scipy = lib.grid(Nodes, Lines, Pros)
net_scipy.solve_pf()
U_scipy = net_scipy.compute_voltages()
v_ref = np.array([abs(u) for u in U_scipy])

# ---------- EVALUACIÓN DE COMBINACIONES ----------

resultados = []

for alpha, rho in tqdm(combinaciones):
    net = lib_itterative_v5.grid(Nodes, Lines, Pros)
    convergido, x_sol, it_fin, estado, hist_eq, hist_ineq, hist_dual = net.solve_iterative(
        alpha=alpha,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        verbose=False
    )

    fila = {
        "alpha": alpha,
        "rho": rho,
        "convergido": convergido,
        "iter": it_fin,
        "res_eq": estado[0],
        "res_ineq": estado[1],
        "res_dual": estado[2]
    }

    # Si se pudo calcular tensiones (aunque no haya convergido), comparamos
    
    try:
        U_iter = net.compute_voltages()
        v_iter = np.array([abs(u) for u in U_iter])
        error_abs = np.linalg.norm(v_iter - v_ref, np.inf)
        error_rel = error_abs / np.linalg.norm(v_ref, np.inf)
        fila["error_abs"] = error_abs
        fila["error_rel"] = error_rel

        # Comparación por pantalla
        
        print(f"\n🔎 Comparación de tensiones solve_iterative vs solve_pf | α={alpha:.1e}, ρ={rho:.1e}")
        print("Nodo |  Iterativo (pu) |  SciPy (pu)  |  Δ Abs     |  Δ Rel (%)")
        for i, (u1, u2) in enumerate(zip(v_iter, v_ref)):
            abs_diff = abs(u1 - u2)
            rel_diff = 100 * abs_diff / abs(u2) if abs(u2) > 0 else 0
            print(f"{i:^4} |  {u1:^15.6f} | {u2:^12.6f} | {abs_diff:^9.2e} | {rel_diff:^9.2f}")
            fila[f"abs_diff_nodo_{i}"] = abs_diff
            fila[f"rel_diff_nodo_{i}"] = rel_diff

        # Comprobación de restricciones cónicas
        
        print("\n¿Se cumplen las restricciones cónicas?")
        for line in net.lines:
            g = line.ineq(net.x)
            print(f"Línea {line.ref}: g(x) = {g:.3e} → {'OK' if g <= 1e-6 else '❌'}")

    except Exception as e:
        print(f"\n❌ No se pudo calcular U_iter para α={alpha:.1e}, ρ={rho:.1e} → {e}")
        fila["error_abs"] = np.nan
        fila["error_rel"] = np.nan

    resultados.append(fila)

# ---------- GUARDADO EN CSV ----------

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_iterativos_comparados.csv", index=False)
print("\n📁 Resultados guardados en 'resultados_iterativos_comparados.csv'")


# ---------- MAPAS DE COLOR con punto mínimo marcado ----------

df_plot = df_resultados[np.isfinite(df_resultados["error_rel"])]

if df_plot.empty:
    print("\n⚠️ No hay combinaciones con error relativo válido para graficar.")
else:
    nodos = [0, 1, 2, 3]
    alphas_unicos = sorted(df_plot["alpha"].unique())
    rhos_unicos = sorted(df_plot["rho"].unique())

    A, R = np.meshgrid(alphas_unicos, rhos_unicos)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for nodo in nodos:
        Z = np.full_like(A, np.nan, dtype=np.float64)

        # Rellenar matriz Z
        for i, rho in enumerate(rhos_unicos):
            for j, alpha in enumerate(alphas_unicos):
                fila = df_plot[(df_plot["alpha"] == alpha) & (df_plot["rho"] == rho)]
                if not fila.empty:
                    val = fila.iloc[0].get(f"rel_diff_nodo_{nodo}")
                    Z[i, j] = val

        ax = axs[nodo]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Error relativo nodo {nodo}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("rho")

        # Mostrar mapa de color o scatter
        if len(alphas_unicos) > 1 and len(rhos_unicos) > 1:
            pcm = ax.pcolormesh(A, R, Z, shading='auto', cmap='viridis')
            fig.colorbar(pcm, ax=ax, label='Error relativo (%)')
        else:
            print(f"ℹ️ Nodo {nodo}: solo hay un valor de alpha o rho → usando scatter.")
            pcm = ax.scatter(A.flatten(), R.flatten(), c=Z.flatten(), cmap='viridis', s=100, edgecolors='k')
            fig.colorbar(pcm, ax=ax, label='Error relativo (%)')

        # Buscar y marcar el mínimo error
        if np.isfinite(Z).any():
            idx_min = np.nanargmin(Z)
            i_min, j_min = np.unravel_index(idx_min, Z.shape)
            alpha_min = alphas_unicos[j_min]
            rho_min = rhos_unicos[i_min]
            z_min = Z[i_min, j_min]

            # Punto rojo
            ax.plot(alpha_min, rho_min, 'ro', markersize=8, label='Mínimo')

            # Texto con alpha, rho y valor
            ax.annotate(f"α={alpha_min:.1e}\nρ={rho_min:.1f}\nErr={z_min:.2f}%",
                        xy=(alpha_min, rho_min),
                        xytext=(10, -10),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", color='red'))

    plt.suptitle("Mapa de error relativo por nodo en función de alpha y rho", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()