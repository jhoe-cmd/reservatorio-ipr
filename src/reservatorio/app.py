import sys
import os
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
from reservatorio.domain.thermal_correction import CorretorTermico
from reservatorio.infrastructure.repositories import JsonCalibrationRepository
from reservatorio.application.optimization import HistoryMatchingService, generate_rmse_surface
from reservatorio.application.montecarlo import MonteCarloIPR
from reservatorio.infrastructure.interface_entrada import InterfaceEntradaDados

# --- IMPORTAÇÃO ACADÊMICA: ÍNDICES DE SOBOL ---
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    salib_disponivel = True
except ImportError:
    salib_disponivel = False

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

if "ghost_curves" not in st.session_state:
    st.session_state["ghost_curves"] = []

st.set_page_config(page_title="Simulador IPR", page_icon="🛢️", layout="wide")

if not salib_disponivel:
    st.error("⚠️ Biblioteca SALib não encontrada. O gráfico de Sobol falhará. Por favor, execute: pip install SALib")

st.title("🛢️ Simulador IPR - Análise de Produtividade")
st.markdown("Plataforma de **History Matching** e **Acoplamento Térmico**.")

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

st.sidebar.markdown("---")
st.sidebar.header("🌡️ Análise de Temperatura (Dissertação)")
ativar_termico = st.sidebar.checkbox("Ativar Acoplamento Térmico", value=True)
if ativar_termico:
    t_ref = st.sidebar.number_input("Temp. Referência PVT (°C)", value=25.0)
    t_res = st.sidebar.number_input("Temp. do Reservatório (°C)", value=60.0)
    incerteza_pct = st.sidebar.slider("Perturbação de Propriedades (%)", -10.0, 10.0, 5.0, step=1.0)
    st.sidebar.caption("Simula a variação térmica em relação ao modelo base.")

if st.sidebar.button("🗑️ Limpar Curvas Comparativas"):
    st.session_state["ghost_curves"] = []
    st.sidebar.success("Histórico limpo!")

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

if st.sidebar.button("Rodar Simulação", type="primary") and salib_disponivel:
    if not dados_validos:
        st.error("Por favor, garanta que os dados da tabela estejam preenchidos.")
    else:
        with st.spinner("Processando otimização e acoplamento térmico..."):
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

                pwf_arr = np.linspace(pe_campo, 0, 50)
                
                if is_fetkovich:
                    q_arr_base = ModelosIPR.fetkovich(pwf_arr, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                    aof_base = ModelosIPR.fetkovich(0.0, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                else:
                    q_arr_base = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)
                    aof_base = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)

                if ativar_termico:
                    j_termico = CorretorTermico.ajustar_indice_J(res_calibracao.J_calibrado, t_res, t_ref, incerteza_pct)
                    if is_fetkovich:
                        q_arr_termico = ModelosIPR.fetkovich(pwf_arr, pe_campo, j_termico, res_calibracao.Psat_calibrado)
                        aof_termico = ModelosIPR.fetkovich(0.0, pe_campo, j_termico, res_calibracao.Psat_calibrado)
                    else:
                        q_arr_termico = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.Psat_calibrado, j_termico)
                        aof_termico = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.Psat_calibrado, j_termico)
                else:
                    aof_termico = aof_base 
                    j_termico = res_calibracao.J_calibrado

                q_arr_plot = q_arr_base * fator_conv
                q_campo_plot = q_campo * fator_conv
                aof_plot = aof_base * fator_conv

                st.session_state["ghost_curves"].append({
                    "name": f"{well_name} ({modelo_escolhido})",
                    "q": q_arr_plot,
                    "pwf": pwf_arr
                })

                fig, ax = plt.subplots(figsize=(11, 5))
                for ghost in st.session_state["ghost_curves"][:-1]:
                    ax.plot(ghost["q"], ghost["pwf"], color='gray', alpha=0.3, linestyle='--', label=f"Histórico: {ghost['name']}")
                
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=3, label=f'IPR Base (AOF: {aof_plot:.0f})')
                
                if ativar_termico:
                    sinal = "+" if incerteza_pct >= 0 else ""
                    ax.plot(q_arr_termico * fator_conv, pwf_arr, color='#e53e3e', linewidth=3, linestyle='--', 
                            label=f'IPR Térmica ({sinal}{incerteza_pct}%) (AOF: {aof_termico*fator_conv:.0f})')
                    ax.fill_betweenx(pwf_arr, q_arr_plot, q_arr_termico * fator_conv, color='#e53e3e', alpha=0.1)

                ax.scatter(q_campo_plot, pwf_campo, color='black', s=60, zorder=5, label='Dados de Teste')
                ax.set_title(f'Desempenho de Fluxo Térmico vs Isotérmico - {well_name}', fontweight='bold', fontsize=12)
                ax.set_xlabel(f'Vazão de Produção ({unidade_vazao})', fontweight='bold')
                ax.set_ylabel('Pressão Dinâmica de Fundo - Pwf (psi)', fontweight='bold')
                ax.set_ylim(0, pe_campo + 500)
                limite_x = max(aof_termico * fator_conv, aof_plot) * 1.1 if ativar_termico else aof_plot * 1.1
                ax.set_xlim(0, limite_x)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', fontsize=9)
                st.pyplot(fig) 

                st.markdown("---")
                st.subheader("🔍 Diagnóstico de Incerteza Numérica e Identificabilidade")

                with st.spinner("Mapeando topografia de erro tridimensional..."):
                    diag = generate_rmse_surface(pwf_campo, q_campo, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado, is_fetkovich)

                    # --- CORREÇÃO DE GRAUS DE LIBERDADE DO QUI-QUADRADO ---
                    N_dados = len(pwf_campo)
                    # Fetkovich tem sempre 2 parâmetros. Darcy-Vogel pode ter 1 (se Psat for travada) ou 2.
                    if is_fetkovich or not travar_psat:
                        chi2_95 = 5.991 # 2 Graus de Liberdade
                        label_dof = "2 g.l."
                    else:
                        chi2_95 = 3.841 # 1 Grau de Liberdade (P2 fixo)
                        label_dof = "1 g.l."
                    
                    fator_expansao = np.sqrt(1.0 + (chi2_95 / N_dados))
                    limiar_incerteza = diag["rmse_min"] * fator_expansao
                    
                    mask_valid = ~np.isnan(diag['RMSE_grid'])
                    area_pixels = np.sum((diag['RMSE_grid'] <= limiar_incerteza) & mask_valid)
                    
                    if np.sum(mask_valid) > 0:
                        diag["area_incerteza_pct"] = (area_pixels / np.sum(mask_valid)) * 100
                    else:
                        diag["area_incerteza_pct"] = 0.0

                    col_diag1, col_diag2 = st.columns(2)
                    with col_diag1:
                        if diag["area_incerteza_pct"] < 5.0:
                            st.success(f"✅ **Região de Confiança ($\chi^2$ {label_dof}):** {diag['area_incerteza_pct']:.1f}% do domínio (Alta Identificabilidade)")
                        elif diag["area_incerteza_pct"] < 20.0:
                            st.warning(f"⚠️ **Região de Confiança ($\chi^2$ {label_dof}):** {diag['area_incerteza_pct']:.1f}% do domínio (Incerteza Moderada)")
                        else:
                            st.error(f"🚨 **Região de Confiança ($\chi^2$ {label_dof}):** {diag['area_incerteza_pct']:.1f}% do domínio (Baixa Identificabilidade)")
                            
                    with col_diag2:
                        # --- CORREÇÃO DA NOMENCLATURA ACADÊMICA ---
                        if np.isnan(diag.get("condicionamento_ci", np.nan)):
                            st.info("ℹ️ **Aspect Ratio (IGA):** Região degenerada. Impossível calcular o alongamento com robustez.")
                        elif diag["condicionamento_ci"] < 10:
                            st.success(f"✅ **Índice Geométrico de Alongamento (IGA):** {diag['condicionamento_ci']:.1f} (Bem-Posto)")
                        elif diag["condicionamento_ci"] < 50:
                            st.warning(f"⚠️ **Índice Geométrico de Alongamento (IGA):** {diag['condicionamento_ci']:.1f} (Túnel de Erro)")
                        else:
                            st.error(f"🚨 **Índice Geométrico de Alongamento (IGA):** {diag['condicionamento_ci']:.1f} (Degenerescência/Vale Estreito)")

                    label_x = 'Coeficiente Performance C' if is_fetkovich else 'Índice de Produtividade J'
                    label_y = 'Expoente de Turbulência n' if is_fetkovich else 'Pressão de Saturação Psat (psi)'

                    fig_3d = go.Figure(data=[go.Surface(z=diag['RMSE_grid'], x=diag['J_grid'], y=diag['Psat_grid'], colorscale='Viridis')])
                    fig_3d.add_trace(go.Scatter3d(x=[res_calibracao.J_calibrado], y=[res_calibracao.Psat_calibrado], z=[diag["rmse_min"]],
                        mode='markers', marker=dict(symbol='diamond', size=7, color='red'), name='Mínimo Global'))
                    fig_3d.update_layout(scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='RMSE (psi)'), height=550)
                    st.plotly_chart(fig_3d, use_container_width=True)

                # --- ANÁLISE GLOBAL DE SENSIBILIDADE DE SOBOL (SALib) ---
                st.markdown("### 🌪️ Índices de Sensibilidade Global de Sobol (Primeira Ordem)")
                
                with st.spinner("Processando Decomposição de Variância via Amostragem de Saltelli..."):
                    # Definição dos limites para amostragem estocástica global (Uniforme)
                    pe_bounds = [pe_campo * 0.85, pe_campo * 1.15] # +/- 15% Incerteza no Reservatório
                    
                    if is_fetkovich:
                        p1_bounds = [max(1e-6, res_calibracao.J_calibrado * 0.5), res_calibracao.J_calibrado * 1.5]
                        p2_bounds = [0.5, 1.0] # Restrição física teórica
                    else:
                        p1_bounds = [max(1e-3, res_calibracao.J_calibrado * 0.5), res_calibracao.J_calibrado * 1.5]
                        p2_bounds = [100.0, pe_campo * 0.999] # Restrição física Darcy-Vogel

                    problem = {
                        'num_vars': 3,
                        'names': ['Pe', 'P1', 'P2'],
                        'bounds': [pe_bounds, p1_bounds, p2_bounds]
                    }

                    # Geração amostral estrita de Saltelli para cálculo iterativo
                    # N=2048 gera 16.384 avaliações da função (N * (2D + 2))
                    param_values = saltelli.sample(problem, 2048)
                    
                    pe_samples = param_values[:, 0]
                    p1_samples = param_values[:, 1]
                    p2_samples = param_values[:, 2]

                    # Cálculo da AOF Vetorizada para cada cenário
                    if is_fetkovich:
                        aof_samples = p1_samples * (pe_samples**2)**p2_samples
                    else:
                        aof_vogel = (p1_samples * (pe_samples - p2_samples)) + ((p1_samples * p2_samples) / 1.8)
                        aof_darcy = p1_samples * pe_samples
                        aof_samples = np.where(pe_samples > p2_samples, aof_vogel, aof_darcy)
                        
                    # Extração rigorosa dos Índices de Primeira Ordem de Sobol (S1)
                    Si = sobol.analyze(problem, aof_samples)
                    
                    # Elimina ruídos numéricos residuais próximos de zero da transformada
                    S1 = np.clip(Si['S1'], 0, None)
                    
                    # Normalização paramétrica percentual real de variação explicada
                    impacto_sobol = (S1 / S1.sum()) * 100

                    labels_tornado = [label_y, 'Pressão Estática Pe', label_x]
                    valores_tornado = [impacto_sobol[2], impacto_sobol[0], impacto_sobol[1]]

                    fig_tornado = go.Figure(go.Bar(
                        x=valores_tornado, y=labels_tornado, orientation='h',
                        marker=dict(color=['#e53e3e' if v > 40 else '#3182ce' for v in valores_tornado]),
                        text=[f"{v:.1f}%" for v in valores_tornado], textposition='auto'
                    ))
                    fig_tornado.update_layout(
                        title="Decomposição da Variância Explicada do Potencial MÁximo (AOF)",
                        xaxis_title="Índice de Sobol $S_1$ (Contribuição Percentual Isolada)",
                        yaxis_title="Parâmetros de Entrada", height=300, margin=dict(l=10, r=10, b=30, t=40)
                    )
                    st.plotly_chart(fig_tornado, use_container_width=True)
                    
                    # Comentário Acadêmico Automático
                    st.caption(f"**Nota Técnica:** A matriz gerou 16.384 cenários simulados. Os Índices de Sobol atestam que **{max(zip(valores_tornado, labels_tornado))[1]}** responde de forma direta e singular por **{max(valores_tornado):.1f}%** de toda a flutuação observada no Potencial de Fluxo Absoluto (AOF).")

                # --- EXPORTAÇÃO BLINDADA ---
                st.markdown("---")
                st.subheader("📥 Geração de Documentação Científica")
                
                str_j = f"{res_calibracao.J_calibrado:.4f}"
                str_p = f"{res_calibracao.Psat_calibrado:.2f}"
                str_pe = f"{pe_campo:.2f}"
                str_rmse = f"{getattr(res_calibracao, 'rmse', 0.0):.2f}"
                str_aof = f"{aof_plot:.2f}"
                
                tex_base = (
                    "\\documentclass{article}\n"
                    "\\usepackage[T1]{fontenc}\n"
                    "\\usepackage[utf8]{inputenc}\n"
                    "\\usepackage{amsmath}\n"
                    "\\begin{document}\n\n"
                    f"\\section*{{Memorial de Calculo Termodinamico - {well_name}}}\n\n"
                    "\\subsection*{1. Ajuste Historico (Modelo Base)}\n"
                    "\\begin{itemize}\n"
                    f"  \\item Erro Residual (RMSE): {str_rmse} psi\n"
                    f"  \\item Parametro 1 Calibrado: {str_j}\n"
                    f"  \\item Parametro 2 Calibrado: {str_p}\n"
                    "\\end{itemize}\n\n"
                )

                if ativar_termico:
                    str_aof_t = f"{(aof_termico * fator_conv):.2f}"
                    str_delta = f"{((aof_termico - aof_base) * fator_conv):+.2f}"
                    str_mult = f"{(j_termico/res_calibracao.J_calibrado):.4f}"
                    tex_termico = (
                        "\\subsection*{2. Acoplamento Termico e Sensibilidade}\n"
                        "\\begin{itemize}\n"
                        f"  \\item Temperatura de Referencia: {t_ref} C\n"
                        f"  \\item Temperatura do Reservatorio: {t_res} C\n"
                        f"  \\item Multiplicador Dinamico de Desempenho: {str_mult}\n"
                        f"  \\item AOF Original (Isotermico): {str_aof} {unidade_vazao}\n"
                        f"  \\item AOF Corrigido (Termico): {str_aof_t} {unidade_vazao}\n"
                        f"  \\item Ganho de Producao Estimado: {str_delta} {unidade_vazao}\n"
                        "\\end{itemize}\n\n"
                    )
                else:
                    tex_termico = f"\\subsection*{{2. Potencial Maximo}}\nO AOF original calculado e de {str_aof} {unidade_vazao}.\n\n"

                latex_content = tex_base + tex_termico + "\\end{document}\n"

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(label="🖩 Baixar Memorial LaTeX", data=latex_content, file_name=f"memorial_{well_name}.tex")
                with col_btn2:
                    df_rel = pd.DataFrame({
                        "Parâmetro": ["Poço", "RMSE", "AOF Base", "AOF Térmico"],
                        "Valor": [well_name, str_rmse, str_aof, f"{(aof_termico * fator_conv):.1f}" if ativar_termico else "-"]
                    })
                    st.download_button(label="📄 Baixar CSV", data=df_rel.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"dados_{well_name}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erro matemático detectado: {e}")