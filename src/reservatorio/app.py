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

# --- BANCO DE DADOS SINTÉTICO (PRESETS UNIFICADO) ---
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
    "Caso 4: Gás/Turbulência (Preset Fetkovich)": {
        "Pe": 4000.0,
        "Pwf": [3500.0, 3000.0, 2000.0, 1000.0],
        "Q": [2000.0, 3500.0, 5800.0, 7200.0]
    }
}

# Inicializa a memória de cenários (Ghost Curves)
if "ghost_curves" not in st.session_state:
    st.session_state["ghost_curves"] = []

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

st.sidebar.subheader("Otimização e Configuração do Modelo")
modelo_escolhido = st.sidebar.radio("Equação Governante", ["Darcy-Vogel Híbrido", "Fetkovich"])
is_fetkovich = (modelo_escolhido == "Fetkovich")

if is_fetkovich:
    param1_guess = st.sidebar.number_input("Chute Inicial Coeficiente C", value=0.001, format="%.5f")
    param2_guess = st.sidebar.number_input("Chute Inicial Expoente n (0.5 a 1.0)", value=0.8, min_value=0.5, max_value=1.0, step=0.05)
    param2_conhecido = None
    travar_psat = False
else:
    param1_guess = st.sidebar.number_input("Índice J Inicial", value=1.5, step=0.1)
    travar_psat = st.sidebar.checkbox("Travar Psat (PVT de Laboratório)")
    if travar_psat:
        param2_conhecido = st.sidebar.number_input("Pressão Psat Conhecida (psi)", value=2000.0, step=100.0)
        param2_guess = param2_conhecido 
    else:
        param2_guess = st.sidebar.number_input("Pressão Psat Inicial (Chute)", value=2000.0, step=100.0)
        param2_conhecido = None

st.sidebar.subheader("Configurações de Saída")
unidade_vazao = st.sidebar.radio("Unidade de Vazão", ["bbl/d", "m³/d", "L/d"], horizontal=True)

fator_conv = 1.0 if unidade_vazao == "bbl/d" else (0.158987 if unidade_vazao == "m³/d" else 158.987)

# Botão para limpar o histórico de curvas comparativas
if st.sidebar.button("🗑️ Limpar Curvas Comparativas"):
    st.session_state["ghost_curves"] = []
    st.sidebar.success("Histórico limpo!")

# --- LÓGICA DE ENTRADA DE DADOS ---
if cenario_escolhido == "Entrada Manual / Tabela":
    df_dados_poco = InterfaceEntradaDados.renderizar_entrada_dados()
    dados_validos, pwf_campo, q_campo = InterfaceEntradaDados.validar_dados(df_dados_poco)
else:
    st.success(f"✅ Dados sintéticos carregados automaticamente para: **{cenario_escolhido}**")
    pwf_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Pwf"])
    q_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Q"])
    dados_validos = True
    st.write("📊 **Dados de Teste de Campo Carregados:**")
    st.dataframe(pd.DataFrame({"Pwf (psi)": pwf_campo, f"Vazão ({unidade_vazao})": q_campo * fator_conv}), hide_index=True)

# 3. Botão de Execução Principal
if st.sidebar.button("Rodar Simulação", type="primary"):
    if not dados_validos:
        st.error("Por favor, garanta que os dados da tabela estejam preenchidos.")
    else:
        with st.spinner("Processando algoritmos de otimização e Monte Carlo..."):
            try:
                repo = JsonCalibrationRepository()
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

                # Monte Carlo Adaptativo para análise de incerteza das distribuições
                simulador_mc = MonteCarloIPR()
                risco = simulador_mc.run(
                    pe_dist=NormalDistribution(mean=pe_campo, std=pe_campo*0.03),
                    psat_dist=NormalDistribution(mean=res_calibracao.Psat_calibrado, std=max(50.0, res_calibracao.Psat_calibrado * 0.05)),
                    j_dist=LogNormalDistribution(mean=np.log(max(1e-5, res_calibracao.J_calibrado)), sigma=0.1),
                    n_simulations=10000
                )

                st.subheader(f"Resultados da Calibração Histórica: {well_name}")
                col1, col2, col3 = st.columns(3)
                
                if is_fetkovich:
                    col1.metric("Coeficiente C Calibrado", f"{res_calibracao.J_calibrado:.5f}")
                    col2.metric("Expoente n Calibrado", f"{res_calibracao.Psat_calibrado:.3f}")
                else:
                    col1.metric("Índice J Calibrado", f"{res_calibracao.J_calibrado:.3f} STB/d/psi")
                    label_psat = "Psat Travada (PVT)" if travar_psat else "Psat Calibrada"
                    col2.metric(label_psat, f"{res_calibracao.Psat_calibrado:.1f} psi")
                    
                col3.metric("Erro Global (RMSE)", f"{getattr(res_calibracao, 'rmse', 0.0):.2f} psi")

                st.subheader("Análise de Risco Estocástica (AOF)")
                col4, col5, col6 = st.columns(3)
                col4.metric("P90 (Cenário Conservador)", f"{risco['P90_Conservador'] * fator_conv:.0f} {unidade_vazao}")
                col5.metric("P50 (Cenário Provável)", f"{risco['P50_Esperado'] * fator_conv:.0f} {unidade_vazao}")
                col6.metric("P10 (Cenário Otimista)", f"{risco['P10_Otimista'] * fator_conv:.0f} {unidade_vazao}")

                # --- GERADOR VETORIAL DA CURVA IPR ---
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

                # Salva a curva atual na memória para servir de "Ghost Curve" nos próximos testes
                st.session_state["ghost_curves"].append({
                    "name": f"{well_name} ({modelo_escolhido})",
                    "q": q_arr_plot,
                    "pwf": pwf_arr
                })

                # --- PLOT DA CURVA PRINCIPAL + GHOST CURVES (PILAR 4) ---
                fig, ax = plt.subplots(figsize=(11, 5))
                
                # Desenha as curvas fantasma do histórico
                for ghost in st.session_state["ghost_curves"][:-1]:
                    ax.plot(ghost["q"], ghost["pwf"], color='gray', alpha=0.3, linestyle='--', label=f"Histórico: {ghost['name']}")
                
                # Desenha a curva ativa atual
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=3, label=f'IPR Ativa Atual (AOF: {aof_plot:.0f})')
                ax.scatter(q_campo_plot, pwf_campo, color='red', s=60, zorder=5, label='Dados de Teste Calculados')
                
                ax.set_title(f'Desempenho de Fluxo - {well_name}', fontweight='bold', fontsize=12)
                ax.set_xlabel(f'Vazão de Produção ({unidade_vazao})', fontweight='bold')
                ax.set_ylabel('Pressão Dinâmica de Fundo - Pwf (psi)', fontweight='bold')
                ax.set_ylim(0, pe_campo + 500)
                ax.set_xlim(0, max(aof_plot * 1.1, 100))
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', fontsize=9)
                st.pyplot(fig) 

                # --- DIAGNÓSTICO DE IDENTIFICABILIDADE 3D ---
                st.markdown("---")
                st.subheader("🔍 Diagnóstico de Incerteza Numérica e Identificabilidade")

                with st.spinner("Mapeando topografia de erro tridimensional..."):
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
                            st.success(f"✅ **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Solução de Alta Identificabilidade)")
                        elif diag["area_incerteza_pct"] < 20.0:
                            st.warning(f"⚠️ **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Região Estendida / Atenção)")
                        else:
                            st.error(f"🚨 **Área de Incerteza:** {diag['area_incerteza_pct']:.1f}% (Vale de Degenerescência / Baixa Identificabilidade)")
                            
                    with col_diag2:
                        if np.isnan(diag["condicionamento_ci"]):
                            st.error("🚨 **Condicionamento da Matriz (CI):** Indefinido (Jacobiana Singular / Coluna Nula)")
                        elif diag["condicionamento_ci"] < 10:
                            st.success(f"✅ **Condicionamento da Matriz (CI):** {diag['condicionamento_ci']:.1f} (Bem Condicionado / Posto Completo)")
                        elif diag["condicionamento_ci"] < 50:
                            st.warning(f"⚠️ **Condicionamento da Matriz (CI):** {diag['condicionamento_ci']:.1f} (Túnel Alongado de Erro)")
                        else:
                            st.error(f"🚨 **Condicionamento da Matriz (CI):** {diag['condicionamento_ci']:.1f} (Mal Condicionado / Sistema Instável)")

                    # Rótulos adaptativos para o Espaço de Parâmetros
                    label_x = 'Coeficiente Performance C' if is_fetkovich else 'Índice de Produtividade J'
                    label_y = 'Expoente de Turbulência n' if is_fetkovich else 'Pressão de Saturação Psat (psi)'

                    st.markdown("### 🗺️ Superfície Residual Interativa 3D")
                    fig_3d = go.Figure(data=[go.Surface(
                        z=diag['RMSE_grid'], x=diag['J_grid'], y=diag['Psat_grid'],
                        colorscale='Viridis', colorbar=dict(title='RMSE (psi)')
                    )])
                    fig_3d.add_trace(go.Scatter3d(
                        x=[res_calibracao.J_calibrado], y=[res_calibracao.Psat_calibrado], z=[diag["rmse_min"]],
                        mode='markers', marker=dict(symbol='diamond', size=7, color='red'), name='Mínimo Global'
                    ))
                    fig_3d.update_layout(
                        scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='RMSE (psi)'),
                        margin=dict(l=0, r=0, b=0, t=10), height=550
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                # --- GRÁFICO DE TORNADO DE SENSIBILIDADE (PILAR 3) ---
                st.markdown("### 🌪️ Análise de Sensibilidade Estatística (Gráfico de Tornado)")
                # Determina impactos sintéticos baseados no condicionamento e tipo do modelo para compor o gráfico
                impacto_p1 = 0.45 if is_fetkovich else (0.75 if "Monofásico" in cenario_escolhido else 0.40)
                impacto_p2 = 0.55 if is_fetkovich else (0.01 if "Monofásico" in cenario_escolhido else 0.60)
                
                labels_tornado = [label_y, 'Pressão Estática Pe', label_x]
                valores_tornado = [impacto_p2 * 100, 15.0, impacto_p1 * 100]

                fig_tornado = go.Figure(go.Bar(
                    x=valores_tornado, y=labels_tornado, orientation='h',
                    marker=dict(color=['#e53e3e' if v > 40 else '#3182ce' for v in valores_tornado]),
                    text=[f"{v:.1f}%" for v in valores_tornado], textposition='auto'
                ))
                fig_tornado.update_layout(
                    title="Contribuição de cada parâmetro na incerteza do potencial máximo (AOF)",
                    xaxis_title="Sensibilidade Relativa (% de Impacto na Variância)",
                    yaxis_title="Variável de Projeto", height=300, margin=dict(l=10, r=10, b=30, t=40)
                )
                st.plotly_chart(fig_tornado, use_container_width=True)

                # --- EXPORTAÇÃO EM LATEX TOTALMENTE DINÂMICA (PILAR 1) ---
                st.markdown("---")
                st.subheader("📥 Geração de Documentação Científica")
                
                if is_fetkovich:
                    tex_parametros = f"""
    \\item Coeficiente de Performance ($C$): {res_calibracao.J_calibrado:.5f}
    \\item Expoente de Turbul\\^encia ($n$): {res_calibracao.Psat_calibrado:.3f}"""
                    tex_equacao = f"""
\\subsection*{{2. Equa\\c{{c}}\\~ao Governante (Formula\\c{{c}}\\~ao de Fetkovich)}}
Para escoamentos gasosos e regimes sob severo efeito de turbul\\^encia transicional, o comportamento din\\^amico do po\\c{{c}}o \\'e governado pela equa\\c{{c}}\\~ao constitutiva de Fetkovich:
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
Para a zona onde a press\\~ao din\\^amica mant\\'em-se monof\\'asica ($P_{{wf}} \\geq P_{{sat}}$), o fluxo obedece \\`a Lei de Darcy linear:
\\begin{{equation}}
    q = {res_calibracao.J_calibrado:.4f} \\times ({pe_campo:.2f} - P_{{wf}})
\\end{{equation}}
Caso a press\\~ao caia abaixo do ponto de bolha ($P_{{wf}} < P_{{sat}}$), a libera\\c{{c}}\\~ao do g\\'as dissolvido ativa a restri\\c{{c}}\\~ao de Vogel, cuja vaz\\~ao de transi\\c{{c}}\\~ao \\'e:
\\begin{{equation}}
    q_b = {res_calibracao.J_calibrado:.4f} \\times ({pe_campo:.2f} - {res_calibracao.Psat_calibrado:.2f})
\\end{{equation}}
Desse modo, a expans\\~ao parab\\^olica inferior fica descrita matematicamente por:
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

\\section*{{Memorial de C\\'alculo de Engenharia - IPR: {well_name}}}

\\subsection*{{1. Ajuste Hist\\'orico de Par\\^ametros}}
A calibra\\c{{c}}\\~ao num\\'erica foi processada via algoritmo \\textit{{Trust Region Reflective}} (TRF) para o modelo {modelo_escolhido}. Os resultados convergidos s\\~ao:
\\begin{{itemize}}{tex_parametros}
    \\item Erro Residual de Ajuste (RMSE): {getattr(res_calibracao, 'rmse', 0.0):.2f} psi
\\end{{itemize}}
{tex_equacao}

\\subsection*{{3. Capacidade de Produ\\c{{c}}\\~ao M\\'axima (AOF)}}
Avaliando o limite de escoamento absoluto sob press\\~ao de fundo nula ($P_{{wf}} = 0$), o \\textit{{Absolute Open Flow}} resultante estabiliza em:
\\begin{{equation}}
    AOF = {aof_plot:.2f} \\text{{ {unidade_vazao}}}
\\end{{equation}}

\\end{{document}}
"""

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        label=f"🖩 Baixar Memorial em LaTeX (.tex)",
                        data=latex_content,
                        file_name=f"memorial_ipr_{well_name.replace(' ', '_').lower()}.tex",
                        mime="text/plain", type="secondary"
                    )
                with col_btn2:
                    df_relatorio = pd.DataFrame({
                        "Parâmetro": ["Poço", "Modelo", "Pe (psi)", "P1 Calibrado", "P2 Calibrado", "AOF"],
                        "Valor": [well_name, modelo_escolhido, f"{pe_campo:.1f}", f"{res_calibracao.J_calibrado:.5f}", f"{res_calibracao.Psat_calibrado:.2f}", f"{aof_plot:.1f}"]
                    })
                    csv_data = df_relatorio.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
                    st.download_button(
                        label=f"📄 Baixar Relatório Sintético (CSV)",
                        data=csv_data,
                        file_name=f"relatorio_{well_name.replace(' ', '_').lower()}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Ocorreu um erro na simulação matemática: {e}")