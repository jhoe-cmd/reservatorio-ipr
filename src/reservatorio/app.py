import sys
import os
# Estas linhas abaixo garantem que o Python ache a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import plotly.graph_objects as go 

# --- IMPORTAÇÕES DA NOSSA NOVA ARQUITETURA ---
from reservatorio.domain.ipr_models import ModelosIPR
from reservatorio.domain.calibration import DarcyVogelCalibration, FetkovichCalibration
from reservatorio.domain.distributions import NormalDistribution, LogNormalDistribution
from reservatorio.infrastructure.repositories import JsonCalibrationRepository
from reservatorio.application.optimization import HistoryMatchingService, generate_rmse_surface
from reservatorio.application.montecarlo import MonteCarloIPR
from reservatorio.infrastructure.interface_entrada import InterfaceEntradaDados

# --- BANCO DE DADOS SINTÉTICO (PRESETS) ---
PRESETS_POCOS = {
    "Entrada Manual / Tabela": None,
    "Caso 1: Pré-Sal (Monofásico - Não Identificável)": {
        "Pe": 6500.0,
        "Pwf": [6000.0, 5500.0, 5000.0, 4500.0],
        "Q": [600.0, 1200.0, 1800.0, 2400.0]
    },
    "Caso 2: Campo Maduro (Bifásico - Vogel)": {
        "Pe": 2500.0,
        "Pwf": [2000.0, 1500.0, 1000.0, 500.0],
        "Q": [980.0, 1780.0, 2380.0, 2780.0]
    },
    "Caso 3: Convencional (Transição Darcy-Vogel)": {
        "Pe": 5000.0,
        "Pwf": [4500.0, 4000.0, 2500.0, 1500.0],
        "Q": [750.0, 1500.0, 3560.0, 4490.0]
    },
    "Caso 4: Gás/Turbulência (Fetkovich)": {
        "Pe": 4000.0,
        "Pwf": [3500.0, 3000.0, 2000.0, 1000.0],
        "Q": [2000.0, 3500.0, 5800.0, 7200.0]
    }
}

# 1. Configuração da Página Web
st.set_page_config(page_title="Simulador IPR", page_icon="🛢️", layout="wide")

st.title("🛢️ Simulador IPR - Análise de Produtividade")
st.markdown("Plataforma de **History Matching** e **Análise de Risco (Monte Carlo)**.")

# 2. Barra Lateral (Inputs do Usuário)
st.sidebar.header("📚 Carregar Cenário")
cenario_escolhido = st.sidebar.selectbox("Selecione um caso de estudo:", list(PRESETS_POCOS.keys()))

st.sidebar.markdown("---")
st.sidebar.header("Parâmetros do Poço")

nome_padrao = cenario_escolhido if cenario_escolhido != "Entrada Manual / Tabela" else "Pré-Sal Santos 01"
well_name = st.sidebar.text_input("Nome do Poço", value=nome_padrao)

pe_default = PRESETS_POCOS[cenario_escolhido]["Pe"] if PRESETS_POCOS[cenario_escolhido] else 6200.0
pe_campo = st.sidebar.number_input("Pressão Estática - Pe (psi)", value=pe_default, step=100.0)

# --- NOVA LÓGICA: SELEÇÃO DE MODELO ---
st.sidebar.subheader("Otimização e Parâmetros")
modelo_escolhido = st.sidebar.radio("Modelo de IPR", ["Darcy-Vogel Híbrido", "Fetkovich"])
is_fetkovich = (modelo_escolhido == "Fetkovich")

if is_fetkovich:
    param1_guess = st.sidebar.number_input("Coeficiente C Inicial", value=0.001, format="%.5f")
    param2_guess = st.sidebar.number_input("Expoente n Inicial (0.5 a 1.0)", value=0.8, min_value=0.5, max_value=1.0, step=0.05)
    param2_conhecido = None
    travar_psat = False
else:
    param1_guess = st.sidebar.number_input("Índice J Inicial", value=1.5, step=0.1)
    travar_psat = st.sidebar.checkbox("Travar Psat (Dado de PVT)")
    if travar_psat:
        param2_conhecido = st.sidebar.number_input("Pressão Psat Conhecida (psi)", value=2000.0, step=100.0)
        param2_guess = param2_conhecido 
    else:
        param2_guess = st.sidebar.number_input("Pressão Psat Inicial (Chute)", value=2000.0, step=100.0)
        param2_conhecido = None

st.sidebar.subheader("Configurações de Saída")
unidade_vazao = st.sidebar.radio("Unidade de Vazão", ["bbl/d", "m³/d", "L/d"], horizontal=True)

if unidade_vazao == "bbl/d":
    fator_conv = 1.0
elif unidade_vazao == "m³/d":
    fator_conv = 0.158987
else: 
    fator_conv = 158.987

# --- LÓGICA DE ENTRADA DE DADOS ---
if cenario_escolhido == "Entrada Manual / Tabela":
    df_dados_poco = InterfaceEntradaDados.renderizar_entrada_dados()
    dados_validos, pwf_campo, q_campo = InterfaceEntradaDados.validar_dados(df_dados_poco)
else:
    st.success(f"✅ Dados sintéticos carregados automaticamente para: **{cenario_escolhido}**")
    pwf_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Pwf"])
    q_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Q"])
    dados_validos = True
    st.write("📊 **Dados de Teste do Cenário:**")
    st.dataframe(pd.DataFrame({"Pwf (psi)": pwf_campo, "Vazão": q_campo}), hide_index=True)

# 3. Botão de Execução
if st.sidebar.button("Rodar Simulação", type="primary"):
    if not dados_validos:
        st.error("Por favor, preencha pelo menos 3 pontos na tabela acima para rodar a calibração.")
    else:
        with st.spinner("Processando algoritmos de otimização..."):
            try:
                repo = JsonCalibrationRepository()
                
                # Seleciona a estratégia dinâmica
                strategy = FetkovichCalibration() if is_fetkovich else DarcyVogelCalibration()
                calibrador = HistoryMatchingService(strategy=strategy, repository=repo)

                res_calibracao = calibrador.calibrar(
                    well_name=well_name,
                    pwf_medidos=pwf_campo,
                    q_medidos=q_campo,
                    Pe=pe_campo,
                    param1_guess=param1_guess,
                    param2_guess=param2_guess,
                    param2_conhecido=param2_conhecido 
                )

                simulador_mc = MonteCarloIPR()
                risco = simulador_mc.run(
                    pe_dist=NormalDistribution(mean=pe_campo, std=pe_campo*0.05),
                    psat_dist=NormalDistribution(mean=res_calibracao.Psat_calibrado, std=abs(res_calibracao.Psat_calibrado*0.1)),
                    j_dist=NormalDistribution(mean=res_calibracao.J_calibrado, std=abs(res_calibracao.J_calibrado*0.1)),
                    n_simulations=50000
                )

                st.subheader(f"Resultados da Calibração: {well_name}")
                col1, col2, col3 = st.columns(3)
                
                # Adapta os textos das métricas ao modelo escolhido
                if is_fetkovich:
                    col1.metric("Coeficiente C", f"{res_calibracao.J_calibrado:.5f}")
                    col2.metric("Expoente n", f"{res_calibracao.Psat_calibrado:.3f}")
                else:
                    col1.metric("Índice J Calibrado", f"{res_calibracao.J_calibrado:.3f} STB/d/psi")
                    label_psat = "Psat Travada (PVT)" if travar_psat else "Psat Calibrada"
                    col2.metric(label_psat, f"{res_calibracao.Psat_calibrado:.1f} psi")
                    
                col3.metric("Erro (RMSE)", f"{getattr(res_calibracao, 'rmse', 0.0):.2f}")

                st.subheader("Análise de Risco Estocástica (AOF)")
                col4, col5, col6 = st.columns(3)
                col4.metric("P90 (Conservador)", f"{risco['P90_Conservador'] * fator_conv:.0f} {unidade_vazao}")
                col5.metric("P50 (Esperado)", f"{risco['P50_Esperado'] * fator_conv:.0f} {unidade_vazao}")
                col6.metric("P10 (Otimista)", f"{risco['P10_Otimista'] * fator_conv:.0f} {unidade_vazao}")

                # --- GERAÇÃO DA CURVA IPR DINÂMICA ---
                pwf_arr = np.linspace(pe_campo, 0, 50)
                
                if is_fetkovich:
                    q_arr = ModelosIPR.fetkovich(pwf_arr, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                    aof = ModelosIPR.fetkovich(0.0, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                else:
                    q_arr = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)
                    aof = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)

                q_arr_plot = q_arr * fator_conv
                q_campo_plot = q_campo * fator_conv
                aof_plot = aof * fator_conv

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=2, label=f'IPR {modelo_escolhido} (AOF: {aof_plot:.0f})')
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

                with st.spinner("Gerando topografia de erro 3D..."):
                    diag = generate_rmse_surface(
                        pwf_medidos=pwf_campo, 
                        q_medidos=q_campo,     
                        Pe=pe_campo,
                        p1_opt=res_calibracao.J_calibrado,
                        p2_opt=res_calibracao.Psat_calibrado,
                        is_fetkovich=is_fetkovich
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

                    # Adapta os eixos do gráfico 3D
                    label_x = 'Coeficiente C' if is_fetkovich else 'Índice J'
                    label_y = 'Expoente n' if is_fetkovich else 'Psat (psi)'

                    st.markdown("### 🗺️ Mapa 3D da Superfície de Erro")
                    fig_3d = go.Figure(data=[go.Surface(
                        z=diag['RMSE_grid'],
                        x=diag['J_grid'],
                        y=diag['Psat_grid'],
                        colorscale='Viridis',
                        colorbar=dict(title='RMSE')
                    )])

                    fig_3d.add_trace(go.Scatter3d(
                        x=[res_calibracao.J_calibrado],
                        y=[res_calibracao.Psat_calibrado],
                        z=[diag["rmse_min"]],
                        mode='markers',
                        marker=dict(symbol='diamond', size=8, color='red'),
                        name='Mínimo Global'
                    ))

                    fig_3d.update_layout(
                        scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='RMSE'),
                        margin=dict(l=0, r=0, b=0, t=30),
                        height=600
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # --- EXPORTAÇÃO EM LATEX DINÂMICA ---
                st.markdown("---")
                st.markdown("#### 📐 Documentação Acadêmica")
                
                # Prepara os blocos condicionais de texto para o LaTeX
                if is_fetkovich:
                    tex_parametros = f"""
    \\item Coeficiente de Performance ($C$): {res_calibracao.J_calibrado:.5f}
    \\item Expoente de Turbul\\^encia ($n$): {res_calibracao.Psat_calibrado:.3f}"""
                    tex_equacao = f"""
\\subsection*{{2. Equa\\c{{c}}\\~ao Governante (Modelo de Fetkovich)}}
Para o escoamento com efeitos de turbul\\^encia e fluxos de g\\'as, o comportamento do po\\c{{c}}o \\'e governado pela equa\\c{{c}}\\~ao generalizada de Fetkovich:
\\begin{{equation}}
    q = {res_calibracao.J_calibrado:.5f} \\times ({pe_campo:.2f}^2 - P_{{wf}}^2)^{{{res_calibracao.Psat_calibrado:.3f}}}
\\end{{equation}}"""
                else:
                    tex_parametros = f"""
    \\item Press\\~ao Est\\'atica ($P_e$): {pe_campo:.2f} psi
    \\item Press\\~ao de Satura\\c{{c}}\\~ao ($P_{{sat}}$): {res_calibracao.Psat_calibrado:.2f} psi
    \\item \\'Indice de Produtividade ($J$): {res_calibracao.J_calibrado:.4f} STB/d/psi"""
                    tex_equacao = f"""
\\subsection*{{2. Equa\\c{{c}}\\~oes Governantes (Modelo H\\'ibrido Darcy-Vogel)}}
Para o regime de fluxo monof\\'asico ($P_{{wf}} \\geq P_{{sat}}$), a vaz\\~ao \\'e dada pela Lei de Darcy:
\\begin{{equation}}
    q = {res_calibracao.J_calibrado:.4f} \\times ({pe_campo:.2f} - P_{{wf}})
\\end{{equation}}
Para o regime bif\\'asico ($P_{{wf}} < P_{{sat}}$), aplica-se o modelo de Vogel. A vaz\\~ao de transi\\c{{c}}\\~ao ($q_b$) \\'e:
\\begin{{equation}}
    q_b = {res_calibracao.J_calibrado:.4f} \\times ({pe_campo:.2f} - {res_calibracao.Psat_calibrado:.2f})
\\end{{equation}}
A curva n\\~ao-linear \\'e governada por:
\\begin{{equation}}
    q = q_b + \\frac{{{res_calibracao.J_calibrado:.4f} \\times {res_calibracao.Psat_calibrado:.2f}}}{{1.8}} \\left[ 1 - 0.2\\left(\\frac{{P_{{wf}}}}{{{res_calibracao.Psat_calibrado:.2f}}}\\right) - 0.8\\left(\\frac{{P_{{wf}}}}{{{res_calibracao.Psat_calibrado:.2f}}}\\right)^2 \\right]
\\end{{equation}}"""

                latex_content = f"""\\documentclass{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[brazil]{{babel}}
\\usepackage{{amsmath}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=2.5cm}}

\\begin{{document}}

\\section*{{Memorial de C\\'alculo - Curva IPR: {well_name}}}

\\subsection*{{1. Par\\^ametros de Entrada e Calibra\\c{{c}}\\~ao}}
Com base no ajuste hist\\'orico utilizando o algoritmo \\textit{{Trust Region Reflective}}, os par\\^ametros otimizados do modelo {modelo_escolhido} s\\~ao:
\\begin{{itemize}}{tex_parametros}
    \\item Erro Residual (RMSE): {getattr(res_calibracao, 'rmse', 0.0):.2f}
\\end{{itemize}}
{tex_equacao}

\\subsection*{{3. Potencial M\\'aximo (AOF)}}
Avaliando o limite te\\'orico para $P_{{wf}} = 0$, o \\textit{{Absolute Open Flow}} calculado \\'e:
\\begin{{equation}}
    AOF = {aof_plot:.2f} \\text{{ {unidade_vazao}}}
\\end{{equation}}

\\end{{document}}
"""

                st.download_button(
                    label=f"🖩 Baixar Memorial em LaTeX (.tex)",
                    data=latex_content,
                    file_name=f"memorial_ipr_{well_name.replace(' ', '_').lower()}.tex",
                    mime="text/plain",
                    type="secondary"
                )

            except Exception as e:
                st.error(f"Ocorreu um erro na simulação matemática: {e}")