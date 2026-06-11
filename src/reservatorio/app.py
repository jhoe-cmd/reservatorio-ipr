import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import chi2

# ==========================
# SALIB (opcional)
# ==========================
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    salib_disponivel = True
except ImportError:
    salib_disponivel = False


# ============================================================
# DOMÍNIO FÍSICO (IPR)
# ============================================================
class ModelosIPR:

    @staticmethod
    def hibrido_darcy_vogel(pwf, pe, psat, j):
        pwf = np.atleast_1d(pwf)
        pwf = np.clip(pwf, 0.0, pe)

        q = np.zeros_like(pwf)

        # Darcy
        mask_d = pwf >= psat
        q[mask_d] = j * (pe - pwf[mask_d])

        # Vogel
        mask_v = pwf < psat
        qb = j * (pe - psat)
        pv = pwf[mask_v] / psat

        q[mask_v] = qb + (j * psat / 1.8) * (1 - 0.2*pv - 0.8*pv**2)

        return np.clip(q, 0, None)

    @staticmethod
    def fetkovich(pwf, pe, c, n):
        pwf = np.atleast_1d(pwf)
        pwf = np.clip(pwf, 0.0, pe)

        dp2 = np.clip(pe**2 - pwf**2, 0, None)
        q = c * (dp2 ** n)

        return np.clip(q, 0, None)


# ============================================================
# CORREÇÃO TÉRMICA
# ============================================================
class CorretorTermico:

    @staticmethod
    def ajustar_j(j, t_res, t_ref, inc):
        fator = 1 + ((t_res - t_ref) / t_ref) * (inc / 100)
        return max(1e-8, j * fator)


# ============================================================
# SSE GRID (IDENTIFICABILIDADE)
# ============================================================
def gerar_sse(pwf, q, pe, j0, psat0, mode):
    grid = 40

    if mode == "fetkovich":
        J, N = np.meshgrid(
            np.linspace(max(1e-6, j0*0.5), j0*1.5, grid),
            np.linspace(0.5, 1.0, grid)
        )
    else:
        J, Psat = np.meshgrid(
            np.linspace(max(1e-3, j0*0.5), j0*1.5, grid),
            np.linspace(100, pe*0.999, grid)
        )

    sse = np.zeros_like(J)

    for i in range(grid):
        for k in range(grid):

            if mode == "fetkovich":
                qcalc = ModelosIPR.fetkovich(pwf, pe, J[i,k], N[i,k])
            else:
                qcalc = ModelosIPR.hibrido_darcy_vogel(pwf, pe, Psat[i,k], J[i,k])

            sse[i,k] = np.sum((qcalc - q)**2)

    return J, Psat if mode != "fetkovich" else N, sse


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="IPR Científico", layout="wide")

st.title("🛢️ Simulador IPR — Versão Científica Corrigida")


# ==========================
# INPUT
# ==========================
pe = st.sidebar.number_input("Pe (psi)", 6500.0)
modelo = st.sidebar.radio("Modelo", ["Darcy-Vogel", "Fetkovich"])
is_fet = (modelo == "Fetkovich")

travar_psat = st.sidebar.checkbox("Travar Psat") if not is_fet else False

tref = st.sidebar.number_input("T ref", 25.0)
tres = st.sidebar.number_input("T res", 60.0)
inc = st.sidebar.slider("Incerteza térmica (%)", -10.0, 10.0, 5.0)

run = st.sidebar.button("Rodar")


# ==========================
# EXECUÇÃO
# ==========================
if run:

    # --------------------------
    # dados sintéticos (mock)
    # --------------------------
    pwf = np.array([6000, 5500, 5000, 4500])
    q = np.array([600, 1200, 1800, 2400])

    # calibração mock
    j = 1.2
    psat = 4000

    if is_fet:
        j = 0.005
        psat = 0.8

    # --------------------------
    # curva base
    # --------------------------
    pwf_grid = np.linspace(pe, 0, 60)

    if is_fet:
        q_base = ModelosIPR.fetkovich(pwf_grid, pe, j, psat)
        aof = ModelosIPR.fetkovich(0, pe, j, psat)
    else:
        q_base = ModelosIPR.hibrido_darcy_vogel(pwf_grid, pe, psat, j)
        aof = ModelosIPR.hibrido_darcy_vogel(0, pe, psat, j)

    # --------------------------
    # térmico
    # --------------------------
    jT = CorretorTermico.ajustar_j(j, tres, tref, inc)

    if is_fet:
        qT = ModelosIPR.fetkovich(pwf_grid, pe, jT, psat)
    else:
        qT = ModelosIPR.hibrido_darcy_vogel(pwf_grid, pe, psat, jT)


    # ==========================
    # PLOT IPR
    # ==========================
    fig, ax = plt.subplots()

    ax.plot(q_base, pwf_grid, label="Base")
    ax.plot(qT, pwf_grid, "--", label="Térmico")

    ax.scatter(q, pwf, c="black", label="Dados")

    ax.set_xlabel("Q")
    ax.set_ylabel("Pwf")
    ax.legend()

    st.pyplot(fig)


    # ==========================
    # SSE + IDENTIFICABILIDADE
    # ==========================
    st.subheader("Identificabilidade")

    Jg, Pg, SSE = gerar_sse(pwf, q, pe, j, psat, modelo.lower())

    sse_min = np.min(SSE)

    chi = chi2.ppf(0.95, df=2)
    lim = sse_min * (1 + chi / len(pwf))

    area = np.mean(SSE <= lim) * 100

    st.write(f"Área de incerteza: {area:.2f}%")


    fig3 = go.Figure(data=[
        go.Surface(z=SSE, x=Jg, y=Pg)
    ])

    st.plotly_chart(fig3, use_container_width=True)


    # ==========================
    # SOBOL (CORRIGIDO)
    # ==========================
    if salib_disponivel:

        st.subheader("Sobol")

        problem = {
            "num_vars": 3,
            "names": ["Pe", "J", "Psat"],
            "bounds": [
                [pe*0.85, pe*1.15],
                [j*0.5, j*1.5],
                [100, pe*0.99]
            ]
        }

        X = saltelli.sample(problem, 512)

        Pe = X[:,0]
        J = X[:,1]
        Ps = X[:,2]

        # 🔴 CORREÇÃO PRINCIPAL:
        # usa curva (não só AOF trivial)
        def qoi(pe, j, ps):
            pwf0 = 0
            if is_fet:
                return ModelosIPR.fetkovich(pwf0, pe, j, ps)
            else:
                return ModelosIPR.hibrido_darcy_vogel(pwf0, pe, ps, j)

        Y = np.array([qoi(a,b,c) for a,b,c in zip(Pe,J,Ps)])

        Si = sobol.analyze(problem, Y)

        st.write("S1:", Si["S1"])
        st.write("ST:", Si["ST"])