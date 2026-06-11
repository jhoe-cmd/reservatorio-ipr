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
# CAMADA DE DOMÍNIO (Física e Termodinâmica Estrita)
# ==============================================================================
class ModelosIPR:
    @staticmethod
    def hibrido_darcy_vogel(pwf, pe, psat, j):
        """Modelo vetorizado N-dimensional com restrições físicas absolutas."""
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
        """Fetkovich vetorizado e blindado matematicamente."""
        pwf = np.asarray(pwf)
        pwf_safe = np.clip(pwf, 0.0, pe)
        
        delta_p_sq = (pe**2) - (pwf_safe**2)
        delta_p_sq = np.clip(delta_p_sq, 0.0, None)
        
        q = c * (delta_p_sq ** n)
        return np.clip(q, 0.0, None)

class CorretorTermico:
    @staticmethod
    def ajustar_indice_J(j_base, t_res, t_ref, ea_r):
        """
        Correção termodinâmica de Arrhenius com base física observável.
        Ea_R: Razão Energia de Ativação / Constante dos Gases (Kelvin).
        """
        tk_res = t_res + 273.15
        tk_ref = t_ref + 273.15
        
        # Arrhenius Estrito para Viscosidade: μ = A * exp(Ea/RT). Logo J ∝ exp(-Ea/RT)
        multiplicador_exponencial = np.exp(-ea_r * ((1.0 / tk_res) - (1.0 / tk_ref)))
        
        return max(1e-8, j_base * multiplicador_exponencial)

# ==============================================================================
# CAMADA DE SERVIÇO (History Matching c/ TRF)
# ==============================================================================
class HistoryMatchingService:
    def calibrar(self, well_name, pwf_medidos, q_medidos, Pe, param1_guess, param2_guess, param2_conhecido, is_fetkovich):
        class ResResultado: pass
        res = ResResultado()
        
        if is_fetkovich:
            def res_func(p):
                return ModelosIPR.fetkovich(pwf=pwf_medidos, pe=Pe, c=p[0], n=p[1]) - q_medidos
            
            opt = least_squares(res_func, [param1_guess, param2_guess], bounds=([1e-6, 0.5], [np.inf, 1.0]), method='trf')
            res.J_calibrado, res.Psat_calibrado = opt.x
            
        else:
            if param2_conhecido is not None:
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf=pwf_medidos, pe=Pe, psat=param2_conhecido, j=p[0]) - q_medidos
                opt = least_squares(res_func, [param1_guess], bounds=([1e-6], [np.inf]), method='trf')
                res.J_calibrado = opt.x[0]
                res.Psat_calibrado = param2_conhecido
            else:
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf=pwf_medidos, pe=Pe, psat=p[1], j=p[0]) - q_medidos
                
                opt = least_squares(res_func, [param1_guess, param2_guess], bounds=([1e-6, 14.7], [np.inf, Pe * 0.999]), method='trf')
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
        
    sse_grid = np.zeros_like(j_grid)
    for i in range(j_grid.shape[0]):
        for k in range(j_grid.shape[1]):
            if is_fetkovich:
                q_calc = ModelosIPR.fetkovich(pwf_medidos, pe, j_grid[i,k], psat_grid[i,k])
            else:
                q_calc = ModelosIPR.hibrido_darcy_vogel(pwf_medidos, pe, psat_grid[i,k], j_grid[i,k])
            sse_grid[i,k] = np.sum((q_calc - q_medidos)**2)
            
    if is_fetkovich:
        q_min = ModelosIPR.fetkovich(pwf_medidos, pe, j_opt, psat_opt)
    else:
        q_min = ModelosIPR.hibrido_darcy_vogel(pwf_medidos, pe, psat_opt, j_opt)
        
    sse_min = np.sum((q_min - q_medidos)**2)
    return j_grid, psat_grid, sse_grid, sse_min

# ==============================================================================
# CAMADA DE APRESENTAÇÃO (Streamlit UI)
# ==============================================================================
PRESETS_POCOS = {
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

st.title("🛢️ Simulador IPR - Física Aplicada & Sensibilidade")
st.markdown("Pipeline analítico com otimização TRF, termodinâmica Arrhenius e decomposição de variância Saltelli.")

st.sidebar.header("📚 Carregar Cenário Experimental")
cenario_escolhido = st.sidebar.selectbox("Preset:", list(PRESETS_POCOS.keys()))
well_name = st.sidebar.text_input("Identificador do Poço", value=cenario_escolhido.split(":")[0])

pe_default = PRESETS_POCOS[cenario_escolhido]["Pe"]
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
st.sidebar.header("🌡️ Parâmetros Termodinâmicos")
ativar_termico = st.sidebar.checkbox("Ativar Acoplamento", value=True)
t_ref = st.sidebar.number_input("T Ref PVT (°C)", value=25.0)
t_res = st.sidebar.number_input("T Reservatório (°C)", value=60.0)
ea_r = st.sidebar.slider("Constante Ea/R de Arrhenius (K)", 500.0, 5000.0, 2000.0, step=100.0, help="Razão entre Energia de Ativação e Constante Universal dos Gases.")

if st.sidebar.button("🗑️ Limpar Gráficos"):
    st.session_state["ghost_curves"] = []

if st.sidebar.button("Rodar Framework Analítico", type="primary") and salib_disponivel:
    with st.spinner("Processando minimização OLS e topografia de verossimilhança..."):
        try:
            pwf_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Pwf"])
            q_campo = np.array(PRESETS_POCOS[cenario_escolhido]["Q"])
            
            # --- 1. HISTORY MATCHING TRF ---
            hm_service = HistoryMatchingService()
            res_calibracao = hm_service.calibrar(
                well_name, pwf_campo, q_campo, pe_campo, 
                param1_guess, param2_guess, param2_conhecido, is_fetkovich
            )

            st.subheader("Resultados da Minimização de Erros (Mínimos Quadrados Não-Lineares)")
            col1, col2, col3 = st.columns(3)
            
            if is_fetkovich:
                col1.metric("C Otimizado", f"{res_calibracao.J_calibrado:.6f}")
                col2.metric("n Otimizado", f"{res_calibracao.Psat_calibrado:.3f}")
            else:
                col1.metric("J Otimizado", f"{res_calibracao.J_calibrado:.4f} STB/d/psi")
                col2.metric("Psat Otimizada", f"{res_calibracao.Psat_calibrado:.1f} psi")
            col3.metric("RMSE Residual", f"{res_calibracao.rmse:.2f} psi")

            # --- 2. CURVA IPR VETORIZADA ---
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
                j_termico = res_calibracao.J_calibrado

            q_arr_plot = q_arr_base * fator_conv
            q_arr_plot_t = q_arr_termico * fator_conv if ativar_termico else q_arr_plot
            
            st.session_state["ghost_curves"].append({"name": f"{well_name}", "q": q_arr_plot, "pwf": pwf_arr})

            fig_ipr, ax = plt.subplots(figsize=(10, 5))
            for ghost in st.session_state["ghost_curves"][:-1]:
                ax.plot(ghost["q"], ghost["pwf"], color='gray', alpha=0.3, linestyle='--')
            
            ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=3, label=f'IPR Base OLS (AOF: {aof_base*fator_conv:.0f})')
            if ativar_termico:
                ax.plot(q_arr_plot_t, pwf_arr, color='#e53e3e', linewidth=3, linestyle='--', label=f'IPR Termodinâmica (AOF: {aof_termico*fator_conv:.0f})')
            ax.scatter(q_campo * fator_conv, pwf_campo, color='black', zorder=5, label='Dados Experimentais')
            ax.set_title("Curvas de Desempenho Físico e Acoplamento de Arrhenius")
            ax.set_xlabel(f"Vazão de Produção ({unidade_vazao})")
            ax.set_ylabel("Pwf Dinâmica (psi)")
            ax.set_ylim(0, pe_campo + 500)
            limite_x = max(aof_termico * fator_conv, aof_base * fator_conv) * 1.1
            ax.set_xlim(0, limite_x)
            ax.grid(True, linestyle=':')
            ax.legend()
            st.pyplot(fig_ipr)

            # --- 3. DIAGNÓSTICO DE IDENTIFICABILIDADE (ESTATÍSTICA F de BATES & WATTS) ---
            st.markdown("---")
            st.subheader("🔍 Região de Confiança Não-Linear de Mínimos Quadrados")
            
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
                sse_limiar = sse_min # Degenerescência de graus de liberdade
                
            mask_valid = ~np.isnan(sse_grid)
            area_pixels = np.sum((sse_grid <= sse_limiar) & mask_valid)
            area_pct = (area_pixels / np.sum(mask_valid)) * 100 if np.sum(mask_valid) > 0 else 0.0

            col_diag1, col_diag2 = st.columns(2)
            col_diag1.metric(f"Região de Confiança (Estatística F-95%)", f"{area_pct:.1f}% do Domínio")
            col_diag2.metric("Mínimo Global SSE", f"{sse_min:.1f} (Resíduo²)")

            label_x = 'Coeficiente C' if is_fetkovich else 'Índice de Produtividade J'
            label_y = 'Expoente n' if is_fetkovich else 'Pressão de Saturação Psat (psi)'

            fig_3d = go.Figure(data=[go.Surface(z=sse_grid, x=j_grid, y=psat_grid, colorscale='Viridis')])
            fig_3d.add_trace(go.Scatter3d(
                x=[res_calibracao.J_calibrado], y=[res_calibracao.Psat_calibrado], z=[sse_min],
                mode='markers', marker=dict(symbol='diamond', size=7, color='red'), name='Ótimo Local'
            ))
            fig_3d.update_layout(scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='SSE'), height=550)
            st.plotly_chart(fig_3d, use_container_width=True)

            # --- 4. SOBOL (INTEGRAL DA IPR PURA - ÁREA DO POTENCIAL) ---
            st.markdown("---")
            st.subheader("🌪️ Análise Global de Sensibilidade de Sobol (Integral da IPR)")
            
            with st.spinner("Processando Integral Numérica e Decomposição de Variância via Saltelli..."):
                pe_bounds = [pe_campo * 0.85, pe_campo * 1.15] 
                if is_fetkovich:
                    p1_bounds = [max(1e-6, res_calibracao.J_calibrado * 0.5), res_calibracao.J_calibrado * 1.5]
                    p2_bounds = [0.5, 1.0] 
                else:
                    p1_bounds = [max(1e-3, res_calibracao.J_calibrado * 0.5), res_calibracao.J_calibrado * 1.5]
                    p2_bounds = [100.0, pe_campo * 0.999] 

                problem = {'num_vars': 3, 'names': ['Pe', 'P1', 'P2'], 'bounds': [pe_bounds, p1_bounds, p2_bounds]}

                # Saltelli gera matriz de N * (2D+2) amostras
                param_values = saltelli.sample(problem, 1024)
                pe_s = param_values[:, 0]
                p1_s = param_values[:, 1]
                p2_s = param_values[:, 2]

                pwf_frac = np.linspace(0, 1, 50) 
                q_integral_samples = []

                # QoI: Área Sob a Curva IPR Pura (Deliverability Potential Area)
                for pe_val, p1_val, p2_val in zip(pe_s, p1_s, p2_s):
                    pe_escalar = float(pe_val)
                    pwf_array = pe_escalar * pwf_frac
                    
                    if is_fetkovich:
                        q_array = ModelosIPR.fetkovich(pwf_array, pe_escalar, float(p1_val), float(p2_val))
                    else:
                        q_array = ModelosIPR.hibrido_darcy_vogel(pwf_array, pe_escalar, float(p2_val), float(p1_val))
                        
                    try:
                        area_ipr = np.trapezoid(q_array, pwf_array)
                    except AttributeError:
                        area_ipr = np.trapz(q_array, pwf_array)
                        
                    q_integral_samples.append(area_ipr)
                    
                q_output = np.array(q_integral_samples)
                Si = sobol.analyze(problem, q_output)
                
                S1 = Si['S1']
                ST = Si['ST']
                
                s1_map = dict(zip(problem['names'], S1))
                st_map = dict(zip(problem['names'], ST))
                
                labels_tornado = [label_y, 'Pressão Estática Pe', label_x]
                s1_plot = [s1_map['P2'], s1_map['Pe'], s1_map['P1']]
                st_plot = [st_map['P2'], st_map['Pe'], st_map['P1']]

                fig_sobol = go.Figure()
                fig_sobol.add_trace(go.Bar(y=labels_tornado, x=s1_plot, orientation='h', name='S1 (Influência Direta)', marker_color='#3182ce'))
                fig_sobol.add_trace(go.Bar(y=labels_tornado, x=st_plot, orientation='h', name='ST (Efeitos Cruzados)', marker_color='#e53e3e'))

                fig_sobol.update_layout(
                    title="Decomposição da Variância da Área de Potencial Absoluto (\u222bIPR)",
                    xaxis_title="Índice de Sobol (Variância Explicada Escalar)",
                    barmode='group', height=400
                )
                st.plotly_chart(fig_sobol, use_container_width=True)
                
                idx_max_s1 = np.argmax(S1)
                nome_max = problem['names'][idx_max_s1]
                nome_formatado = label_y if nome_max == 'P2' else (label_x if nome_max == 'P1' else 'Pressão Estática Pe')
                
                st.caption(f"**Nota Científica:** O Modelo avaliou a Integral Estrita da curva de influxo para {len(q_output)} amostras. O parâmetro **{nome_formatado}** controla o sistema estatisticamente, com índice $S_1$ indicando responsabilidade isolada de **{S1[idx_max_s1]*100:.1f}%** da Área de Potencial Absoluto.")

            # --- EXPORTAÇÃO BLINDADA ---
            st.markdown("---")
            st.subheader("📥 Geração de Relatório Físico-Estatístico")
            
            str_j = f"{res_calibracao.J_calibrado:.4f}"
            str_p = f"{res_calibracao.Psat_calibrado:.2f}"
            str_pe = f"{pe_campo:.2f}"
            str_rmse = f"{getattr(res_calibracao, 'rmse', 0.0):.2f}"
            str_aof = f"{aof_plot:.2f}"
            
            tex_base = (
                "\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath}\n\\begin{document}\n\n"
                f"\\section*{{Relat\\'orio Cient\\'ifico de Produtividade - Po\\c{{c}}o: {well_name}}}\n\n"
                "\\subsection*{1. Calibra\\c{{c}}\\~ao via M\\'inimos Quadrados N\\~ao-Lineares}\n\\begin{itemize}\n"
                f"  \\item Erro Residual (RMSE): {str_rmse} psi\n"
                f"  \\item Parametro 1 Calibrado: {str_j}\n"
                f"  \\item Parametro 2 Calibrado: {str_p}\n\\end{itemize}\n\n"
            )

            if ativar_termico:
                str_aof_t = f"{(aof_termico * fator_conv):.2f}"
                str_mult = f"{(j_termico/res_calibracao.J_calibrado):.4f}"
                tex_termico = (
                    "\\subsection*{2. Acoplamento Termodin\\^amico (Lei de Arrhenius)}\n\\begin{itemize}\n"
                    f"  \\item Temp. Refer\\^encia: {t_ref} C\n"
                    f"  \\item Temp. Reservat\\'orio: {t_res} C\n"
                    f"  \\item Raz\\~ao de Ativa\\c{{c}}\\~ao (Ea/R): {ea_r} K\n"
                    f"  \\item Multiplicador Exponencial F\\'isico: {str_mult}\n"
                    f"  \\item AOF Base: {str_aof} {unidade_vazao}\n"
                    f"  \\item AOF T\\'ermico: {str_aof_t} {unidade_vazao}\n\\end{itemize}\n\n"
                )
            else:
                tex_termico = f"\\subsection*{{2. Potencial M\\'aximo}}\nAOF estabilizado em {str_aof} {unidade_vazao}.\n\n"

            tex_sobol = (
                "\\subsection*{3. Sensibilidade Global de Sobol (\\textit{Quantity of Interest}: $\\int q \\ dP_{wf}$)}\n"
                "\\begin{itemize}\n"
                f"  \\item $S_1$ (Press\\~ao Est\\'atica): {S1[0]:.4f} | $S_T$: {ST[0]:.4f}\n"
                f"  \\item $S_1$ (Coeficiente/\\'Indice): {S1[1]:.4f} | $S_T$: {ST[1]:.4f}\n"
                f"  \\item $S_1$ (Expoente/Psat): {S1[2]:.4f} | $S_T$: {ST[2]:.4f}\n\\end{itemize}\n\n"
            )

            latex_content = tex_base + tex_termico + tex_sobol + "\\end{document}\n"

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.download_button(label="🖩 Baixar Memorial LaTeX", data=latex_content, file_name=f"memorial_{well_name}.tex")
            with col_btn2:
                df_rel = pd.DataFrame({"Parâmetro": ["Poço", "RMSE", "AOF Base", "AOF Térmico", "Sobol S1(Pe)", "Sobol S1(P1)", "Sobol S1(P2)"],
                                       "Valor": [well_name, str_rmse, str_aof, f"{(aof_termico * fator_conv):.1f}" if ativar_termico else "-", f"{S1[0]:.4f}", f"{S1[1]:.4f}", f"{S1[2]:.4f}"]})
                st.download_button(label="📄 Baixar CSV Analítico", data=df_rel.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"dados_{well_name}.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erro Crítico Matemático: {e}")