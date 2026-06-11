import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import scipy.stats as stats
from scipy.optimize import least_squares

# --- IMPORTAÇÃO ACADÊMICA: ÍNDICES DE SOBOL ---
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    salib_disponivel = True
except ImportError:
    salib_disponivel = False

# ==============================================================================
# CAMADA DE DOMÍNIO (Física Vetorial e Tensorial)
# ==============================================================================
class ModelosIPR:
    @staticmethod
    def hibrido_darcy_vogel(pwf, pe, psat, j):
        pwf = np.asarray(pwf)
        q = np.zeros_like(pwf, dtype=float)
        
        pwf_safe = np.clip(pwf, 0.0, pe)
        
        mask_darcy = pwf_safe >= psat
        q = np.where(mask_darcy, j * (pe - pwf_safe), q)
        
        mask_vogel = pwf_safe < psat
        qb = j * (pe - psat)
        pwf_v = pwf_safe
        
        q_vogel = qb + (j * psat / 1.8) * (1.0 - 0.2*(pwf_v/psat) - 0.8*(pwf_v/psat)**2)
        q = np.where(mask_vogel, q_vogel, q)
        
        return np.clip(q, 0.0, None)

    @staticmethod
    def fetkovich(pwf, pe, c, n):
        pwf = np.asarray(pwf)
        pwf_safe = np.clip(pwf, 0.0, pe)
        
        delta_p_sq = (pe**2) - (pwf_safe**2)
        delta_p_sq = np.clip(delta_p_sq, 0.0, None)
        
        q = c * (delta_p_sq ** n)
        return np.clip(q, 0.0, None)

class CorretorTermico:
    @staticmethod
    def ajustar_indice_J(j_base, t_res, t_ref, ea_r):
        tk_res = t_res + 273.15
        tk_ref = t_ref + 273.15
        multiplicador_exponencial = np.exp(-ea_r * ((1.0 / tk_res) - (1.0 / tk_ref)))
        return max(1e-8, j_base * multiplicador_exponencial)

# ==============================================================================
# CAMADA DE SERVIÇO E INFERÊNCIA ESTATÍSTICA
# ==============================================================================
class HistoryMatchingService:
    def calibrar(self, well_name, pwf_medidos, q_medidos, Pe, param1_guess, param2_guess, param2_conhecido, is_fetkovich):
        class ResResultado: pass
        res = ResResultado()
        
        # --- CORREÇÃO DE BOUNDS DO CHUTE INICIAL (Prevenção do Erro Crítico) ---
        if is_fetkovich:
            # Clamping do chute para garantir que esteja dentro de [0.5, 1.0] e J > 1e-6
            p1_start = max(1e-6, param1_guess)
            p2_start = np.clip(param2_guess, 0.5, 1.0)
            
            def res_func(p):
                return ModelosIPR.fetkovich(pwf_medidos, Pe, p[0], p[1]) - q_medidos
                
            opt = least_squares(res_func, [p1_start, p2_start], bounds=([1e-6, 0.5], [np.inf, 1.0]), method='trf')
            res.J_calibrado, res.Psat_calibrado = opt.x
            
        else:
            p1_start = max(1e-6, param1_guess)
            
            if param2_conhecido is not None:
                # Se Psat for travada e o usuário informou um valor fora da física, nós o "empurramos" para o limite
                psat_fixa = min(param2_conhecido, Pe * 0.99)
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf_medidos, Pe, psat_fixa, p[0]) - q_medidos
                    
                opt = least_squares(res_func, [p1_start], bounds=([1e-6], [np.inf]), method='trf')
                res.J_calibrado = opt.x[0]
                res.Psat_calibrado = psat_fixa
            else:
                # Clamping do chute inicial da Psat para garantir que esteja entre [14.7, 99% da Pe]
                p2_start = np.clip(param2_guess, 14.7, Pe * 0.99)
                
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf_medidos, Pe, p[1], p[0]) - q_medidos
                    
                opt = least_squares(res_func, [p1_start, p2_start], bounds=([1e-6, 14.7], [np.inf, Pe * 0.999]), method='trf')
                res.J_calibrado, res.Psat_calibrado = opt.x
                
        res.rmse = np.sqrt(np.mean(opt.fun**2))
        return res

@st.cache_data
def calcular_sse_matriz_exata(pwf_medidos, q_medidos, pe, j_opt, psat_opt, is_fetkovich):
    res_malha = 100j 
    
    if is_fetkovich:
        j_grid, psat_grid = np.mgrid[max(1e-6, j_opt*0.2):j_opt*2.0:res_malha, 0.5:1.0:res_malha]
    else:
        j_grid, psat_grid = np.mgrid[max(1e-3, j_opt*0.2):j_opt*2.0:res_malha, 100.0:(pe*0.999):res_malha]
        
    pwf_brd = pwf_medidos[:, np.newaxis, np.newaxis]
    q_medidos_brd = q_medidos[:, np.newaxis, np.newaxis]
    
    if is_fetkovich:
        q_calc_tensor = ModelosIPR.fetkovich(pwf_brd, pe, j_grid, psat_grid)
    else:
        q_calc_tensor = ModelosIPR.hibrido_darcy_vogel(pwf_brd, pe, psat_grid, j_grid)
        
    sse_grid = np.sum((q_calc_tensor - q_medidos_brd)**2, axis=0)
            
    if is_fetkovich:
        q_min = ModelosIPR.fetkovich(pwf_medidos, pe, j_opt, psat_opt)
    else:
        q_min = ModelosIPR.hibrido_darcy_vogel(pwf_medidos, pe, psat_opt, j_opt)
        
    sse_min = np.sum((q_min - q_medidos)**2)
    return j_grid, psat_grid, sse_grid, sse_min

# ==============================================================================
# CAMADA DE APRESENTAÇÃO E INTEGRAÇÃO DE ENTRADA DE DADOS
# ==============================================================================
class InterfaceEntradaDadosMock:
    @staticmethod
    def renderizar_entrada_dados():
        st.info("Para usar a Tabela de Entrada interativa, certifique-se de que a classe `InterfaceEntradaDados` original está sendo chamada corretamente.")
        return pd.DataFrame()
    @staticmethod
    def validar_dados(df):
        return False, None, None

try:
    from reservatorio.infrastructure.interface_entrada import InterfaceEntradaDados
except ImportError:
    InterfaceEntradaDados = InterfaceEntradaDadosMock

PRESETS_POCOS = {
    "Entrada Manual / Tabela": None,
    "Caso 1: Pré-Sal (Monofásico - Não Identificável)": {
        "Pe": 6500.0, "Pwf": [6000.0, 5500.0, 5000.0, 4500.0], "Q": [600.0, 1200.0, 1800.0, 2400.0]
    },
    "Caso 2: Campo Maduro (Bifásico - Vogel)": {
        "Pe": 2500.0, "Pwf": [2000.0, 1500.0, 1000.0, 500.0], "Q": [980.0, 1780.0, 2380.0, 2780.0]
    },
    "Caso 3: Convencional (Transição Darcy-Vogel)": {
        "Pe": 5000.0, "Pwf": [4500.0, 4000.0, 2500.0, 1500.0], "Q": [750.0, 1500.0, 3560.0, 4490.0]
    }
}

if "ghost_curves" not in st.session_state:
    st.session_state["ghost_curves"] = []
st.session_state["ghost_curves"] = st.session_state["ghost_curves"][-5:]

st.set_page_config(page_title="Simulador IPR Científico", page_icon="🛢️", layout="wide")

if not salib_disponivel:
    st.error("⚠️ Biblioteca SALib não encontrada. O gráfico de Sobol falhará. Execute no terminal: pip install SALib")

st.title("🛢️ Simulador IPR - Física Aplicada & Sensibilidade Térmica")
st.markdown("Pipeline analítico com otimização TRF e decomposição de variância do campo térmico.")

st.sidebar.header("📚 Carregar Cenário Experimental")
cenario_escolhido = st.sidebar.selectbox("Preset:", list(PRESETS_POCOS.keys()))

nome_padrao = cenario_escolhido.split(":")[0] if cenario_escolhido != "Entrada Manual / Tabela" else "Poço-Pre-Sal-01"
well_name = st.sidebar.text_input("Identificador do Poço", value=nome_padrao)

pe_default = PRESETS_POCOS[cenario_escolhido]["Pe"] if PRESETS_POCOS[cenario_escolhido] else 6200.0
pe_campo = st.sidebar.number_input("Pressão Estática Pe (psi)", value=pe_default, step=100.0)

modelo_escolhido = st.sidebar.radio("Modelo Físico", ["Darcy-Vogel Híbrido", "Fetkovich"])
is_fetkovich = (modelo_escolhido == "Fetkovich")

if is_fetkovich:
    param1_guess = st.sidebar.number_input("Chute C (Coeficiente)", value=0.001, format="%.5f")
    param2_guess = st.sidebar.number_input("Chute n (Expoente)", value=0.8, min_value=0.5, max_value=1.0)
    param2_conhecido = None
    travar_psat = False
else:
    param1_guess = st.sidebar.number_input("Chute J (Índice)", value=1.5, step=0.1)
    travar_psat = st.sidebar.checkbox("Fixar Psat via PVT")
    if travar_psat:
        param2_conhecido = st.sidebar.number_input("Psat Lab (psi)", value=3000.0, step=100.0)
        param2_guess = param2_conhecido 
    else:
        param2_guess = st.sidebar.number_input("Chute Psat (psi)", value=3000.0, step=100.0)
        param2_conhecido = None

unidade_vazao = st.sidebar.radio("Unidade de Vazão", ["bbl/d", "m³/d"], horizontal=True)
fator_conv = 1.0 if unidade_vazao == "bbl/d" else 0.158987

st.sidebar.markdown("---")
st.sidebar.header("🌡️ Campo de Temperatura (Dissertação Elias)")
ativar_termico = st.sidebar.checkbox("Ativar Acoplamento Forward", value=True)
t_ref = st.sidebar.number_input("T Ref PVT (°C)", value=25.0)
t_res = st.sidebar.number_input("T Reservatório (°C)", value=60.0)
ea_r = st.sidebar.slider("Constante Aparente (Ea/R) em K", 500.0, 5000.0, 2000.0, step=100.0)

st.sidebar.markdown("---")
st.sidebar.header("🌪️ Sensibilidade Estocástica do Campo Térmico")
var_sobol_pct = st.sidebar.slider(
    "Incerteza Paramétrica Térmica (%)", 
    min_value=1.0, max_value=20.0, value=5.0, step=1.0, 
    help="Janela de ruído nos sensores de temperatura e calibração de laboratório (T_res, T_ref, Ea/R)."
)

if st.sidebar.button("🗑️ Limpar Gráficos"):
    st.session_state["ghost_curves"] = []

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


if st.sidebar.button("Rodar Framework Analítico", type="primary") and salib_disponivel:
    if not dados_validos:
        st.error("Por favor, preencha a tabela com pelo menos 3 pontos numéricos válidos antes de rodar a simulação.")
    else:
        with st.spinner("Resolvendo Sistema Tensorial e Estatístico..."):
            try:
                # --- 1. HISTORY MATCHING TRF ---
                hm_service = HistoryMatchingService()
                res_calibracao = hm_service.calibrar(
                    well_name, pwf_campo, q_campo, pe_campo, 
                    param1_guess, param2_guess, param2_conhecido, is_fetkovich
                )

                st.subheader("Resultados da Minimização de Erros OLS")
                col1, col2, col3 = st.columns(3)
                
                if is_fetkovich:
                    col1.metric("C Base", f"{res_calibracao.J_calibrado:.6f}")
                    col2.metric("n Otimizado", f"{res_calibracao.Psat_calibrado:.3f}")
                else:
                    col1.metric("J Base", f"{res_calibracao.J_calibrado:.4f} STB/d/psi")
                    col2.metric("Psat Otimizada", f"{res_calibracao.Psat_calibrado:.1f} psi")
                col3.metric("RMSE Residual", f"{res_calibracao.rmse:.2f} psi")

                # --- 2. CURVA IPR E FORWARD TÉRMICO ---
                pwf_arr = np.linspace(pe_campo, 0, 100)
                
                if is_fetkovich:
                    q_arr_base = ModelosIPR.fetkovich(pwf_arr, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                    aof_base = ModelosIPR.fetkovich(0.0, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado)
                else:
                    q_arr_base = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)
                    aof_base = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.Psat_calibrado, res_calibracao.J_calibrado)

                if ativar_termico:
                    j_termico = CorretorTermico.ajustar_indice_J(res_calibracao.J_calibrado, t_res, t_ref, ea_r)
                    if is_fetkovich:
                        q_arr_termico = ModelosIPR.fetkovich(pwf_arr, pe_campo, j_termico, res_calibracao.Psat_calibrado)
                        aof_termico = ModelosIPR.fetkovich(0.0, pe_campo, j_termico, res_calibracao.Psat_calibrado)
                    else:
                        q_arr_termico = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.Psat_calibrado, j_termico)
                        aof_termico = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.Psat_calibrado, j_termico)
                else:
                    aof_termico = aof_base 

                q_arr_plot = q_arr_base * fator_conv
                aof_plot = aof_base * fator_conv
                q_arr_plot_t = q_arr_termico * fator_conv if ativar_termico else q_arr_plot
                
                st.session_state["ghost_curves"].append({"name": f"{well_name}", "q": q_arr_plot, "pwf": pwf_arr})

                fig_ipr, ax = plt.subplots(figsize=(10, 5))
                for ghost in st.session_state["ghost_curves"][:-1]:
                    ax.plot(ghost["q"], ghost["pwf"], color='gray', alpha=0.3, linestyle='--')
                
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=3, label=f'IPR Base OLS (AOF: {aof_plot:.0f})')
                if ativar_termico:
                    ax.plot(q_arr_plot_t, pwf_arr, color='#e53e3e', linewidth=3, linestyle='--', label=f'IPR Predição Térmica (AOF: {aof_termico*fator_conv:.0f})')
                ax.scatter(q_campo * fator_conv, pwf_campo, color='black', zorder=5, label='Dados Experimentais')
                ax.set_title("Curvas de Desempenho e Acoplamento Preditivo de Arrhenius")
                ax.set_xlabel(f"Vazão de Produção ({unidade_vazao})")
                ax.set_ylabel("Pwf Dinâmica (psi)")
                ax.set_ylim(0, pe_campo + 500)
                limite_x = max(aof_termico * fator_conv, aof_plot) * 1.1
                ax.set_xlim(0, limite_x)
                ax.grid(True, linestyle=':')
                ax.legend()
                st.pyplot(fig_ipr)

                # --- 3. DIAGNÓSTICO DE IDENTIFICABILIDADE VETORIZADO ---
                st.markdown("---")
                st.subheader("🔍 Espaço Paramétrico e Topografia de Verossimilhança")
                
                j_grid, psat_grid, sse_grid, sse_min = calcular_sse_matriz_exata(
                    pwf_campo, q_campo, pe_campo, res_calibracao.J_calibrado, res_calibracao.Psat_calibrado, is_fetkovich
                )

                N_dados = len(pwf_campo)
                p_livres = 2 if (is_fetkovich or not travar_psat) else 1
                v_df = N_dados - p_livres
                
                if v_df > 0:
                    f_95 = stats.f.ppf(0.95, dfn=p_livres, dfd=v_df)
                    sse_limiar = sse_min * (1.0 + (float(p_livres) / v_df) * f_95)
                else:
                    sse_limiar = sse_min 
                    
                mask_valid = ~np.isnan(sse_grid)
                area_pixels = np.sum((sse_grid <= sse_limiar) & mask_valid)
                area_pct = (area_pixels / np.sum(mask_valid)) * 100 if np.sum(mask_valid) > 0 else 0.0

                col_diag1, col_diag2 = st.columns(2)
                col_diag1.metric(f"Região de Confiança Rigorosa (F-95%)", f"{area_pct:.1f}% do Domínio")
                col_diag2.metric("Mínimo Global OLS", f"{sse_min:.1f} (SSE)")

                label_x = 'C' if is_fetkovich else 'J'
                label_y = 'n' if is_fetkovich else 'Psat'

                fig_3d = go.Figure(data=[go.Surface(z=sse_grid, x=j_grid, y=psat_grid, colorscale='Viridis')])
                fig_3d.add_trace(go.Scatter3d(
                    x=[res_calibracao.J_calibrado], y=[res_calibracao.Psat_calibrado], z=[sse_min],
                    mode='markers', marker=dict(symbol='diamond', size=7, color='red'), name='Mínimo Estrito'
                ))
                fig_3d.update_layout(scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='SSE'), height=550)
                st.plotly_chart(fig_3d, use_container_width=True)

                # --- 4. SOBOL - FOCADO EXCLUSIVAMENTE NO CAMPO TÉRMICO ---
                if ativar_termico:
                    st.markdown("---")
                    st.subheader(f"🌪️ Análise de Sensibilidade Global do Campo Térmico (\u00B1{var_sobol_pct}%)")
                    
                    with st.spinner("Avaliando propagação de incerteza da termodinâmica de Arrhenius..."):
                        delta = var_sobol_pct / 100.0
                        
                        bounds_t_res = [max(0.1, t_res * (1.0 - delta)), t_res * (1.0 + delta)]
                        bounds_t_ref = [max(0.1, t_ref * (1.0 - delta)), t_ref * (1.0 + delta)]
                        bounds_ea_r  = [max(1.0, ea_r * (1.0 - delta)), ea_r * (1.0 + delta)]

                        problem = {
                            'num_vars': 3, 
                            'names': ['T_Reservatorio', 'T_Referencia', 'Ea/R'], 
                            'bounds': [bounds_t_res, bounds_t_ref, bounds_ea_r]
                        }
                        
                        param_values = saltelli.sample(problem, 1024)
                        
                        t_res_s = param_values[:, 0]
                        t_ref_s = param_values[:, 1]
                        ea_r_s  = param_values[:, 2]

                        pwf_frac = np.linspace(0, 1, 50) 
                        pwf_array_fixo = pe_campo * pwf_frac
                        
                        integral_termica_samples = []

                        for t_res_val, t_ref_val, ea_r_val in zip(t_res_s, t_ref_s, ea_r_s):
                            j_pertubado = CorretorTermico.ajustar_indice_J(
                                res_calibracao.J_calibrado, float(t_res_val), float(t_ref_val), float(ea_r_val)
                            )
                            
                            if is_fetkovich:
                                q_array = ModelosIPR.fetkovich(pwf_array_fixo, pe_campo, j_pertubado, res_calibracao.Psat_calibrado)
                            else:
                                q_array = ModelosIPR.hibrido_darcy_vogel(pwf_array_fixo, pe_campo, res_calibracao.Psat_calibrado, j_pertubado)
                            
                            try:
                                integral_ipr = np.trapezoid(q_array, pwf_array_fixo)
                            except AttributeError:
                                integral_ipr = np.trapz(q_array, pwf_array_fixo)
                                
                            integral_termica_samples.append(integral_ipr)
                            
                        q_output = np.array(integral_termica_samples)
                        Si = sobol.analyze(problem, q_output)
                        
                        S1 = Si['S1']
                        ST = Si['ST']
                        
                        s1_map = dict(zip(problem['names'], S1))
                        st_map = dict(zip(problem['names'], ST))
                        
                        labels_tornado = ['Razão Arrhenius (Ea/R)', 'T. Referência (PVT)', 'T. Reservatório']
                        s1_plot = [s1_map['Ea/R'], s1_map['T_Referencia'], s1_map['T_Reservatorio']]
                        st_plot = [st_map['Ea/R'], st_map['T_Referencia'], st_map['T_Reservatorio']]

                        fig_sobol = go.Figure()
                        fig_sobol.add_trace(go.Bar(y=labels_tornado, x=s1_plot, orientation='h', name='S1 (Impacto Isolado)', marker_color='#3182ce'))
                        fig_sobol.add_trace(go.Bar(y=labels_tornado, x=st_plot, orientation='h', name='ST (Impacto Total c/ Interação)', marker_color='#e53e3e'))

                        fig_sobol.update_layout(
                            title=f"Influência das Variáveis Térmicas na Produtividade (Janela \u00B1{var_sobol_pct}%)",
                            xaxis_title="Índice de Sobol (Fração da Incerteza Explicada)",
                            barmode='group', height=400
                        )
                        st.plotly_chart(fig_sobol, use_container_width=True)
                        
                        idx_max_s1 = np.argmax(S1)
                        nome_max = problem['names'][idx_max_s1]
                        
                        st.caption(f"**Análise da Dissertação:** Para isolar a influência termo-física do reservatório, os parâmetros isotérmicos base ($J$ e $P_{{sat}}$) foram cravados no ótimo OLS. O Sobol perturbou as {len(q_output)} amostras variando exclusivamente as variáveis do campo de temperatura. O resultado indica que **{nome_max}** é o fator termodinâmico dominante, controlando **{S1[idx_max_s1]*100:.1f}%** do desvio preditivo da produção.")
                else:
                    st.info("Ative o 'Acoplamento Forward' na barra lateral para habilitar a Análise de Sobol do campo térmico.")

                # --- EXPORTAÇÃO BLINDADA EM LATEX ---
                st.markdown("---")
                st.subheader("📥 Geração de Relatório Físico-Estatístico")
                
                str_j = f"{res_calibracao.J_calibrado:.4f}"
                str_p = f"{res_calibracao.Psat_calibrado:.2f}"
                str_pe = f"{pe_campo:.2f}"
                str_rmse = f"{getattr(res_calibracao, 'rmse', 0.0):.2f}"
                str_aof = f"{aof_plot:.2f}"
                
                tex_base = f"""\\documentclass{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\begin{{document}}

\\section*{{Memorial F\\'isico-Estat\\'istico - Po\\c{{c}}o: {well_name}}}

\\subsection*{{1. Formula\\c{{c}}\\~ao do Problema Inverso OLS}}
\\begin{{itemize}}
  \\item RMSE Residual do Truncamento: {str_rmse} psi
  \\item Parametro 1 (J/C) Convergido: {str_j}
  \\item Parametro 2 (Psat/n) Convergido: {str_p}
\\end{{itemize}}

"""

                if ativar_termico:
                    str_aof_t = f"{(aof_termico * fator_conv):.2f}"
                    str_mult = f"{(j_termico/res_calibracao.J_calibrado):.4f}"
                    str_delta = f"{((aof_termico - aof_base) * fator_conv):+.2f}"
                    tex_termico = f"""\\subsection*{{2. Forward Preditivo Termodin\\^amico de Arrhenius}}
\\begin{{itemize}}
  \\item Delta de Temperatura: {t_ref}C a {t_res}C
  \\item Raz\\~ao Aparente Constante F\\'isica (Ea/R): {ea_r} K
  \\item Tensor Exponencial de Fluxo: {str_mult}
  \\item AOF Base: {str_aof} {unidade_vazao}
  \\item AOF T\\'ermico: {str_aof_t} {unidade_vazao}
  \\item Delta AOF Absoluto: {str_delta} {unidade_vazao}
\\end{{itemize}}

\\subsection*{{3. Sensibilidade Global Termodin\\^amica (Janela Estoc\\'astica $\\pm{var_sobol_pct}\\%$)}}
\\begin{{itemize}}
  \\item $S_1$ (Temp. Reservatorio): {S1[0]:.4f} | $S_T$: {ST[0]:.4f}
  \\item $S_1$ (Temp. Referencia): {S1[1]:.4f} | $S_T$: {ST[1]:.4f}
  \\item $S_1$ (Razao Ea/R): {S1[2]:.4f} | $S_T$: {ST[2]:.4f}
\\end{{itemize}}
"""
                else:
                    tex_termico = f"""\\subsection*{{2. Potencial M\\'aximo OLS}}
AOF estabilizado: {str_aof} {unidade_vazao}.

"""

                latex_content = tex_base + tex_termico + "\\end{document}\n"

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(label="🖩 Baixar Memorial LaTeX", data=latex_content, file_name=f"memorial_{well_name}.tex")
                with col_btn2:
                    df_rel = pd.DataFrame({"Parâmetro": ["Poço", "RMSE", "AOF Base", "AOF Térmico", "Sobol S1(T_res)", "Sobol S1(T_ref)", "Sobol S1(Ea_R)"],
                                           "Valor": [well_name, str_rmse, str_aof, f"{(aof_termico * fator_conv):.1f}" if ativar_termico else "-", f"{S1[0]:.4f}" if ativar_termico else "-", f"{S1[1]:.4f}" if ativar_termico else "-", f"{S1[2]:.4f}" if ativar_termico else "-"]})
                    st.download_button(label="📄 Baixar Tensor Numérico CSV", data=df_rel.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"dados_{well_name}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erro Crítico Matemático: {e}")