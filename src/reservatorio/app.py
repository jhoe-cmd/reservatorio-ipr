import sys
import os
# Estas linhas abaixo garantem que o Python ache a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from reservatorio.domain.ipr_models import DarcyVogelHibridoIPR
from reservatorio.domain.calibration import DarcyVogelCalibration
from reservatorio.domain.distributions import NormalDistribution, LogNormalDistribution
from reservatorio.infrastructure.repositories import JsonCalibrationRepository
from reservatorio.application.optimization import HistoryMatchingService
from reservatorio.application.montecarlo import MonteCarloIPR

# 1. Configuração da Página Web
st.set_page_config(page_title="Simulador IPR", page_icon="🛢️", layout="wide")

st.title("🛢️ Simulador IPR - Análise de Produtividade")
st.markdown("Plataforma de **History Matching** e **Análise de Risco (Monte Carlo)**.")

# 2. Barra Lateral (Inputs do Usuário)
st.sidebar.header("Parâmetros do Poço")
well_name = st.sidebar.text_input("Nome do Poço", value="Poço Alpha")
pe_campo = st.sidebar.number_input("Pressão Estática - Pe (psi)", value=4000.0, step=100.0)

st.sidebar.subheader("Dados de Teste (Separador)")
pwf_str = st.sidebar.text_input("Pressões de Fundo (Pwf)", value="3500, 3000, 2500, 1500")
q_str = st.sidebar.text_input("Vazões Correspondentes (Q)", value="800, 1550, 2200, 3100")

st.sidebar.subheader("Chutes Iniciais (Otimização)")
j_guess = st.sidebar.number_input("Índice J Inicial", value=1.5, step=0.1)
psat_guess = st.sidebar.number_input("Pressão Psat Inicial", value=2000.0, step=100.0)

# 3. Botão de Execução
if st.sidebar.button("Rodar Simulação", type="primary"):
    with st.spinner("Processando algoritmos de otimização e Monte Carlo..."):
        try:
            # Processamento dos dados de entrada
            pwf_campo = np.array([float(x.strip()) for x in pwf_str.split(',')])
            q_campo = np.array([float(x.strip()) for x in q_str.split(',')])
            
            if len(pwf_campo) != len(q_campo):
                st.error("Erro: A quantidade de pressões e vazões deve ser exatamente igual!")
                st.stop()

            # Instanciação dos serviços
            repo = JsonCalibrationRepository()
            calibrador = HistoryMatchingService(strategy=DarcyVogelCalibration(), repository=repo)

            # Execução da Calibração
            res_calibracao = calibrador.calibrar(
                well_name=well_name,
                pwf_medidos=pwf_campo,
                q_medidos=q_campo,
                Pe=pe_campo,
                J_guess=j_guess,
                Psat_guess=psat_guess
            )

            # Execução do Monte Carlo
            simulador_mc = MonteCarloIPR()
            risco = simulador_mc.run(
                pe_dist=NormalDistribution(mean=pe_campo, std=pe_campo*0.05),
                psat_dist=NormalDistribution(mean=res_calibracao.Psat_calibrado, std=150.0),
                j_dist=LogNormalDistribution(mean=np.log(res_calibracao.J_calibrado), sigma=0.15),
                n_simulations=50000
            )

            # 4. Apresentação Visual dos Resultados
            st.subheader(f"Resultados da Calibração: {well_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Índice J Calibrado", f"{res_calibracao.J_calibrado:.3f} STB/d/psi")
            col2.metric("Psat Calibrada", f"{res_calibracao.Psat_calibrado:.1f} psi")
            col3.metric("Erro (RMSE)", f"{getattr(res_calibracao, 'rmse', 0.0):.2f}")

            st.subheader("Análise de Risco Estocástica (AOF)")
            col4, col5, col6 = st.columns(3)
            col4.metric("P90 (Conservador)", f"{risco['P90_Conservador']:.0f} bbl/d")
            col5.metric("P50 (Esperado)", f"{risco['P50_Esperado']:.0f} bbl/d")
            col6.metric("P10 (Otimista)", f"{risco['P10_Otimista']:.0f} bbl/d")

            # Plotagem do Gráfico na Interface
            modelo = DarcyVogelHibridoIPR()
            class MockPoco:
                Pe = pe_campo
                Psat = res_calibracao.Psat_calibrado
                q_test = q_campo[1] if len(q_campo) > 1 else q_campo[0]
                Pwf_test = pwf_campo[1] if len(pwf_campo) > 1 else pwf_campo[0]
                
            q_arr, pwf_arr, _, aof = modelo.calcular_curva(MockPoco(), J_in=res_calibracao.J_calibrado)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(q_arr, pwf_arr, 'b-', linewidth=2, label=f'IPR Calibrada (AOF: {aof:.0f})')
            ax.scatter(q_campo, pwf_campo, color='red', zorder=5, label='Dados de Teste')
            ax.set_title(f'Curva IPR - {well_name}', fontweight='bold')
            ax.set_xlabel('Vazão (STB/dia)', fontweight='bold')
            ax.set_ylabel('Pressão de Fundo - Pwf (psi)', fontweight='bold')
            ax.set_ylim(0, pe_campo + 500)
            ax.set_xlim(0, aof * 1.1)
            ax.grid(True, linestyle='--')
            ax.legend()
            
            st.pyplot(fig) # Comando mágico do Streamlit para exibir o gráfico Matplotlib

        except Exception as e:
            st.error(f"Ocorreu um erro na simulação matemática: {e}")