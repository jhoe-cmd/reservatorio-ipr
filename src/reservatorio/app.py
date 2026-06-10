import sys
import os
# Estas linhas abaixo garantem que o Python ache a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import plotly.graph_objects as go # <-- Importação do Plotly para o 3D

# --- IMPORTAÇÕES DA NOSSA NOVA ARQUITETURA ---
from reservatorio.domain.ipr_models import ModelosIPR
from reservatorio.domain.calibration import DarcyVogelCalibration
from reservatorio.domain.distributions import NormalDistribution, LogNormalDistribution
from reservatorio.infrastructure.repositories import JsonCalibrationRepository
from reservatorio.application.optimization import HistoryMatchingService, generate_rmse_surface
from reservatorio.application.montecarlo import MonteCarloIPR
from reservatorio.infrastructure.interface_entrada import InterfaceEntradaDados # Nova Tabela

# 1. Configuração da Página Web
st.set_page_config(page_title="Simulador IPR", page_icon="🛢️", layout="wide")

st.title("🛢️ Simulador IPR - Análise de Produtividade")
st.markdown("Plataforma de **History Matching** e **Análise de Risco (Monte Carlo)**.")

# 2. Barra Lateral (Inputs do Usuário)
st.sidebar.header("Parâmetros do Poço")
well_name = st.sidebar.text_input("Nome do Poço", value="Pré-Sal Santos 01")
pe_campo = st.sidebar.number_input("Pressão Estática - Pe (psi)", value=6200.0, step=100.0)

st.sidebar.subheader("Otimização e Parâmetros")
j_guess = st.sidebar.number_input("Índice J Inicial", value=1.5, step=0.1)

travar_psat = st.sidebar.checkbox("Travar Psat (Dado de Laboratório/PVT)")
if travar_psat:
    psat_conhecida = st.sidebar.number_input("Pressão Psat Conhecida (psi)", value=4500.0, step=100.0)
    psat_guess = psat_conhecida 
else:
    psat_guess = st.sidebar.number_input("Pressão Psat Inicial (Chute)", value=2000.0, step=100.0)
    psat_conhecida = None

st.sidebar.subheader("Configurações de Saída")
unidade_vazao = st.sidebar.radio("Unidade de Vazão", ["bbl/d", "m³/d", "L/d"], horizontal=True)

if unidade_vazao == "bbl/d":
    fator_conv = 1.0
elif unidade_vazao == "m³/d":
    fator_conv = 0.158987
else: 
    fator_conv = 158.987

# --- NOVA INTERFACE DE ENTRADA (UPLOAD / EXCEL) ---
df_dados_poco = InterfaceEntradaDados.renderizar_entrada_dados()
dados_validos, pwf_campo, q_campo = InterfaceEntradaDados.validar_dados(df_dados_poco)

# 3. Botão de Execução
if st.sidebar.button("Rodar Simulação", type="primary"):
    if not dados_validos:
        st.error("Por favor, preencha pelo menos 3 pontos na tabela acima para rodar a calibração.")
    else:
        with st.spinner("Processando algoritmos de otimização e Monte Carlo..."):
            try:
                # O split de strings sumiu, os dados já vêm validados do Pandas!
                
                repo = JsonCalibrationRepository()
                calibrador = HistoryMatchingService(strategy=DarcyVogelCalibration(), repository=repo)

                res_calibracao = calibrador.calibrar(
                    well_name=well_name,
                    pwf_medidos=pwf_campo,
                    q_medidos=q_campo,
                    Pe=pe_campo,
                    J_guess=j_guess,
                    Psat_guess=psat_guess,
                    Psat_conhecida=psat_conhecida 
                )

                simulador_mc = MonteCarloIPR()
                risco = simulador_mc.run(
                    pe_dist=NormalDistribution(mean=pe_campo, std=pe_campo*0.05),
                    psat_dist=NormalDistribution(mean=res_calibracao.Psat_calibrado, std=150.0),
                    j_dist=LogNormalDistribution(mean=np.log(res_calibracao.J_calibrado), sigma=0.15),
                    n_simulations=50000
                )

                st.subheader(f"Resultados da Calibração: {well_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Índice J Calibrado", f"{res_calibracao.J_calibrado:.3f} STB/d/psi")
                
                label_psat = "Psat Travada (PVT)" if travar_psat else "Psat Calibrada"
                col2.metric(label_psat, f"{res_calibracao.Psat_calibrado:.1f} psi")
                col3.metric("Erro (RMSE)", f"{getattr(res_calibracao, 'rmse', 0.0):.2f}")

                st.subheader("Análise de Risco Estocástica (AOF)")
                col4, col5, col6 = st.columns(3)
                col4.metric("P90 (Conservador)", f"{risco['P90_Conservador'] * fator_conv:.0f} {unidade_vazao}")
                col5.metric("P50 (Esperado)", f"{risco['P50_Esperado'] * fator_conv:.0f} {unidade_vazao}")
                col6.metric("P10 (Otimista)", f"{risco['P10_Otimista'] * fator_conv:.0f} {unidade_vazao}")

                # --- GERAÇÃO DA CURVA IPR (CÓDIGO OTIMIZADO) ---
                # 1. Criar um array de pressões (de Pe até 0) para traçar a curva IPR suave
                pwf_arr = np.linspace(pe_campo, 0, 50)
                
                # 2. Calcular as vazões teóricas usando a nossa nova Camada de Domínio
                q_arr = ModelosIPR.hibrido_darcy_vogel(
                    pwf=pwf_arr, 
                    pe=pe_campo, 
                    psat=res_calibracao.Psat_calibrado, 
                    j=res_calibracao.J_calibrado
                )
                
                # 3. Calcular o AOF (Potencial Máximo) passando Pwf = 0
                aof = ModelosIPR.hibrido_darcy_vogel(
                    pwf=0.0, 
                    pe=pe_campo, 
                    psat=res_calibracao.Psat_calibrado, 
                    j=res_calibracao.J_calibrado
                )

                q_arr_plot = q_arr * fator_conv
                q_campo_plot = q_campo * fator_conv
                aof_plot = aof * fator_conv

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=2, label=f'IPR Calibrada (AOF: {aof_plot:.0f})')
                ax.scatter(q_campo_plot, pwf_campo, color='red', zorder=5, label='Dados de Teste')
                ax.set_title(f'Curva IPR - {well_name}', fontweight='bold')
                ax.set_xlabel(f'Vazão ({unidade_vazao})', fontweight='bold')
                ax.set_ylabel('Pressão de Fundo - Pwf (psi)', fontweight='bold')
                ax.set_ylim(0, pe_campo + 500)
                ax.set_xlim(0, aof_plot * 1.1)
                ax.grid(True, linestyle='--')
                ax.legend()
                
                st.pyplot(fig) 

                st.markdown("---")
                st.subheader("🔍 Diagnóstico de Incerteza e Identificabilidade")

                with st.spinner("Gerando topografia de erro e analisando condicionamento..."):
                    diag = generate_rmse_surface(
                        pwf_medidos=pwf_campo, 
                        q_medidos=q_campo,     
                        Pe=pe_campo,
                        J_opt=res_calibracao.J_calibrado,
                        Psat_opt=res_calibracao.Psat_calibrado
                    )

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

                    # --- NOVA VISUALIZAÇÃO 3D INTERATIVA (PLOTLY) ---
                    st.markdown("### 🗺️ Mapa 3D da Superfície de Erro")
                    st.info("💡 Dica: Arraste com o mouse para girar o gráfico e visualizar o 'túnel' de não-identificabilidade.")
                    
                    fig_3d = go.Figure(data=[go.Surface(
                        z=diag['RMSE_grid'],
                        x=diag['J_grid'],
                        y=diag['Psat_grid'],
                        colorscale='Viridis',
                        colorbar=dict(title='RMSE (psi)')
                    )])

                    # Adiciona o ponto ótimo (a estrela) no gráfico 3D
                    fig_3d.add_trace(go.Scatter3d(
                        x=[res_calibracao.J_calibrado],
                        y=[res_calibracao.Psat_calibrado],
                        z=[diag["rmse_min"]],
                        mode='markers',
                        marker=dict(symbol='diamond', size=8, color='red'),
                        name='Mínimo Global'
                    ))

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title='Índice J',
                            yaxis_title='Psat (psi)',
                            zaxis_title='RMSE Residual'
                        ),
                        margin=dict(l=0, r=0, b=0, t=30),
                        height=600
                    )
                    
                    # Renderiza no Streamlit
                    st.plotly_chart(fig_3d, use_container_width=True)

                # --- NOVA SEÇÃO: FASE 3 - EXPORTAÇÃO DE RELATÓRIO ---
                st.markdown("---")
                st.subheader("📥 Exportar Resultados")
                
                # 1. Estruturando os dados em um dicionário
                dados_relatorio = {
                    "Parâmetro": [
                        "Nome do Poço",
                        "Pressão Estática (Pe) [psi]",
                        "Índice de Produtividade (J) [STB/d/psi]",
                        "Pressão de Saturação (Psat) [psi]",
                        "Status da Psat",
                        "Erro da Calibração (RMSE) [psi]",
                        f"AOF (Potencial Máximo) [{unidade_vazao}]",
                        f"P90 (Conservador) [{unidade_vazao}]",
                        f"P50 (Esperado) [{unidade_vazao}]",
                        f"P10 (Otimista) [{unidade_vazao}]",
                        "Área de Incerteza [%]",
                        "Condicionamento (CI)"
                    ],
                    "Valor": [
                        well_name,
                        f"{pe_campo:.2f}",
                        f"{res_calibracao.J_calibrado:.4f}",
                        f"{res_calibracao.Psat_calibrado:.2f}",
                        "Travada (Laboratório/PVT)" if travar_psat else "Calibrada Numericamente",
                        f"{diag['rmse_min']:.2f}",
                        f"{aof_plot:.2f}",
                        f"{risco['P90_Conservador'] * fator_conv:.2f}",
                        f"{risco['P50_Esperado'] * fator_conv:.2f}",
                        f"{risco['P10_Otimista'] * fator_conv:.2f}",
                        f"{diag['area_incerteza_pct']:.2f}",
                        f"{diag['condicionamento_ci']:.2f}" if not np.isnan(diag['condicionamento_ci']) else "Indefinido"
                    ]
                }

                # 2. Convertendo para DataFrame Pandas
                df_relatorio = pd.DataFrame(dados_relatorio)

                # 3. Gerando o CSV com codificação pt-BR (excel brasileiro)
                csv = df_relatorio.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

                # 4. Botão de Download na Interface
                st.download_button(
                    label=f"📄 Baixar Relatório - {well_name} (CSV)",
                    data=csv,
                    file_name=f"relatorio_ipr_{well_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    type="primary"
                )

            except Exception as e:
                st.error(f"Ocorreu um erro na simulação matemática: {e}")