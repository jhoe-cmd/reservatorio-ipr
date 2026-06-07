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
from reservatorio.application.optimization import HistoryMatchingService, generate_rmse_surface
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
            
            st.pyplot(fig) 

            # --- NOVA SEÇÃO: DIAGNÓSTICO DE INCERTEZA ---
            st.markdown("---")
            st.subheader("🔍 Diagnóstico de Incerteza e Identificabilidade")

            with st.spinner("Gerando topografia de erro e analisando condicionamento..."):
                # 1. Chama a nova função de diagnóstico (com as variáveis corretas)
                diag = generate_rmse_surface(
                    pwf_medidos=pwf_campo, 
                    q_medidos=q_campo,     
                    Pe=pe_campo,
                    J_opt=res_calibracao.J_calibrado,
                    Psat_opt=res_calibracao.Psat_calibrado
                )

                # 2. Painel Automático de Saúde da Calibração
                col_diag1, col_diag2 = st.columns(2)
                
                with col_diag1:
                    if diag["area_incerteza_pct"] < 5.0:
                        st.success(f"✅ **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Solução Robusta)")
                    elif diag["area_incerteza_pct"] < 20.0:
                        st.warning(f"⚠️ **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Atenção)")
                    else:
                        st.error(f"🚨 **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Baixa Identificabilidade)")
                        
                with col_diag2:
                    if np.isnan(diag["condicionamento_ci"]):
                        st.error("🚨 **Condicionamento (CI):** Indefinido (Matriz singular/sem dados válidos)")
                    elif diag["condicionamento_ci"] < 10:
                        st.success(f"✅ **Condicionamento (CI):** {diag['condicionamento_ci']:.1f} (Bem condicionado)")
                    elif diag["condicionamento_ci"] < 50:
                        st.warning(f"⚠️ **Condicionamento (CI):** {diag['condicionamento_ci']:.1f} (Vale alongado)")
                    else:
                        st.error(f"🚨 **Condicionamento (CI):** {diag['condicionamento_ci']:.1f} (Mal condicionado)")

                # 3. Plotagem do Mapa de Contorno RMSE
                fig_map, ax_map = plt.subplots(figsize=(8, 6))

                cp = ax_map.contourf(
                    diag['J_grid'], diag['Psat_grid'], diag['RMSE_grid'], 
                    levels=30, cmap='viridis_r', extend='max'
                )
                fig_map.colorbar(cp, label='RMSE (psi)')

                ax_map.contour(
                    diag['J_grid'], diag['Psat_grid'], diag['RMSE_grid'], 
                    levels=[diag['limiar_incerteza']], colors='red', linewidths=2, linestyles='dashed'
                )

                ax_map.scatter(
                    [res_calibracao.J_calibrado], [res_calibracao.Psat_calibrado], 
                    marker='*', color='white', s=300, edgecolors='black', 
                    label=f'Ótimo (RMSE: {diag["rmse_min"]:.1f} psi)'
                )

                ax_map.set_title("Superfície de Erro: $RMSE = f(J, P_{sat})$", fontweight='bold')
                ax_map.set_xlabel('Índice de Produtividade - J (STB/d/psi)', fontweight='bold')
                ax_map.set_ylabel('Pressão de Saturação - Psat (psi)', fontweight='bold')
                ax_map.legend()
                ax_map.grid(True, linestyle=':', alpha=0.6)

                st.pyplot(fig_map)

        except Exception as e:
            st.error(f"Ocorreu um erro na simulação matemática: {e}")