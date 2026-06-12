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
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    salib_disponivel = True
except ImportError:
    try:
        from SALib.sample import saltelli as sobol_sample
        from SALib.analyze import sobol
        salib_disponivel = True
    except:
        salib_disponivel = False

# ==============================================================================
# CAMADA DE DOMÍNIO (Física Vetorial e Tensorial 100% Vetorizada)
# ==============================================================================
class ModelosIPR:
    @staticmethod
    def hibrido_darcy_vogel(pwf, pe, psat, j):
        pwf = np.asarray(pwf, dtype=np.float32)
        shape = np.broadcast(pwf, pe, psat, j).shape
        q = np.zeros(shape, dtype=np.float32)
        
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
        pwf = np.asarray(pwf, dtype=np.float32)
        shape = np.broadcast(pwf, pe, c, n).shape
        pwf_safe = np.clip(pwf, 0.0, pe)
        
        delta_p_sq = (pe**2) - (pwf_safe**2)
        delta_p_sq = np.clip(delta_p_sq, 0.0, None)
        
        q = c * (delta_p_sq ** n)
        return np.broadcast_to(np.clip(q, 0.0, None), shape)

class CorretorTermico:
    @staticmethod
    def calcular_razao_viscosidade(t_res, t_ref, ea_r):
        """
        Sensibilidade Térmica Semi-Acoplada (First-Order Mobility Approximation):
        Arrhenius perturba estritamente a viscosidade mu. Bo, Rs e kro mantidos estáticos.
        """
        tk_res = t_res + 273.15
        tk_ref = t_ref + 273.15
        return np.exp(ea_r * ((1.0 / tk_ref) - (1.0 / tk_res)))

    @staticmethod
    def ajustar_indice_J(j_base, t_res, t_ref, ea_r):
        razao_mu = CorretorTermico.calcular_razao_viscosidade(t_res, t_ref, ea_r)
        return np.maximum(1e-8, j_base * razao_mu)

    @staticmethod
    def ajustar_Psat(psat_base, t_res, t_ref, correlacao="Standing"):
        t_ref_f = (t_ref * 1.8) + 32.0
        t_res_f = (t_res * 1.8) + 32.0
        
        if correlacao == "Standing":
            psat_nova = (psat_base + 25.48) * (10 ** (0.00091 * (t_res_f - t_ref_f))) - 25.48
            return np.maximum(14.7, psat_nova)
            
        elif correlacao == "Vazquez-Beggs":
            t_ref_r = t_ref_f + 460.0
            t_res_r = t_res_f + 460.0
            psat_nova = psat_base * np.exp(-705.586 * ((1.0 / t_res_r) - (1.0 / t_ref_r)))
            return np.maximum(14.7, psat_nova)
            
        elif correlacao == "Glaso":
            a = -0.30218
            b = 1.7447
            c = 1.7669 - np.log10(psat_base)
            delta = b**2 - 4*a*c
            
            delta_safe = np.maximum(delta, 0.0)
            
            # Avaliação rigorosa das raízes para evitar ramos não-físicos da inversão de Glaso
            root1 = (-b + np.sqrt(delta_safe)) / (2 * a)
            root2 = (-b - np.sqrt(delta_safe)) / (2 * a)
            x_ref = np.where(np.abs(root1) < np.abs(root2), root1, root2)
            
            x_res = x_ref + 0.172 * np.log10(t_res_f / t_ref_f)
            log_psat_nova = 1.7669 + 1.7447 * x_res + a * (x_res ** 2)
            psat_nova = np.where(delta < 0, psat_base, 10 ** log_psat_nova)
            return np.maximum(14.7, psat_nova)
            
        elif correlacao == "Petrosky-Farshad":
            k = 4.561e-5
            shift = k * ((t_res_f ** 1.3911) - (t_ref_f ** 1.3911))
            psat_nova = psat_base * (10 ** shift)
            return np.maximum(14.7, psat_nova)
            
        return psat_base

# ==============================================================================
# CAMADA DE SERVIÇO E INFERÊNCIA ESTATÍSTICA (OLS, Covariância, AIC, Resíduos)
# ==============================================================================
class HistoryMatchingService:
    def calibrar(self, well_name, pwf_medidos, q_medidos, Pe, param1_guess, param2_guess, param2_conhecido, is_fetkovich):
        class ResResultado: pass
        res = ResResultado()
        res.psat_auto_locked = False
        
        N = len(pwf_medidos)
        
        if is_fetkovich:
            res.k_params = 2
            p1_start = max(1e-8, param1_guess)
            p2_start = np.clip(param2_guess, 0.5, 1.5)
            
            def res_func(p):
                return ModelosIPR.fetkovich(pwf_medidos, Pe, p[0], p[1]) - q_medidos
                
            opt = least_squares(res_func, [p1_start, p2_start], bounds=([1e-8, 0.2], [np.inf, 2.0]), method='trf')
            res.param1_calibrado, res.param2_calibrado = opt.x
            
            res.residuos = opt.fun
            ss_res_raw = np.sum(opt.fun**2)
            
        else:
            p1_start = max(1e-6, param1_guess)
            
            # Critério de Drawdown para proteção da Jacobiana
            pwf_min = np.min(pwf_medidos)
            drawdown_relativo = (Pe - pwf_min) / Pe
            is_strict_monophasic = drawdown_relativo <= 0.10
            
            if param2_conhecido is not None:
                res.k_params = 1
                psat_fixa = min(param2_conhecido, Pe * 0.99)
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf_medidos, Pe, psat_fixa, p[0]) - q_medidos
                    
                opt = least_squares(res_func, [p1_start], bounds=([1e-6], [np.inf]), method='trf')
                res.param1_calibrado = opt.x[0]
                res.param2_calibrado = psat_fixa
                
            elif is_strict_monophasic:
                res.psat_auto_locked = True
                res.k_params = 1
                psat_fixa = min(param2_guess, Pe * 0.99)
                
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf_medidos, Pe, psat_fixa, p[0]) - q_medidos
                    
                opt = least_squares(res_func, [p1_start], bounds=([1e-6], [np.inf]), method='trf')
                res.param1_calibrado = opt.x[0]
                res.param2_calibrado = psat_fixa
            else:
                res.k_params = 2
                p2_start = np.clip(param2_guess, 14.7, Pe * 0.99)
                def res_func(p):
                    return ModelosIPR.hibrido_darcy_vogel(pwf_medidos, Pe, p[1], p[0]) - q_medidos
                    
                opt = least_squares(res_func, [p1_start, p2_start], bounds=([1e-6, 14.7], [np.inf, Pe * 0.999]), method='trf')
                res.param1_calibrado, res.param2_calibrado = opt.x
            
            res.residuos = opt.fun
            ss_res_raw = np.sum(opt.fun**2)
                
        ss_res = max(ss_res_raw, 1e-12)
        ss_tot = np.sum((q_medidos - np.mean(q_medidos))**2)
        
        res.rmse = np.sqrt(ss_res / N)
        
        if ss_tot > 0:
            res.r2 = 1.0 - (ss_res / ss_tot)
            # R2 Ajustado
            if N - res.k_params - 1 > 0:
                res.r2_adj = 1.0 - ((1.0 - res.r2) * (N - 1) / (N - res.k_params - 1))
            else:
                res.r2_adj = np.nan
        else:
            res.r2 = np.nan
            res.r2_adj = np.nan
        
        den = np.sum(np.abs(q_medidos))
        res.wmape = (np.sum(np.abs(opt.fun))/den) * 100.0 if den > 0 else np.nan

        res.aic = N * np.log(ss_res / N) + 2 * res.k_params
        if N - res.k_params - 1 > 0:
            res.aicc = res.aic + (2 * res.k_params * (res.k_params + 1)) / (N - res.k_params - 1)
        else:
            res.aicc = np.nan
            
        res.cov_mat = None
        res.corr_mat = None
        res.cond_number = np.nan
        res.cov_degenerada = False
        res.ci_p1 = np.nan
        res.ci_p2 = np.nan
        
        v_df = max(1, N - res.k_params)
        sigma2 = ss_res / v_df
        
        try:
            J_jac = opt.jac
            hess_approx = J_jac.T @ J_jac
            res.cond_number = np.linalg.cond(hess_approx)
            
            H_inv = np.linalg.pinv(hess_approx)
            res.cov_mat = sigma2 * H_inv
            
            d = np.sqrt(np.maximum(np.diag(res.cov_mat), 1e-15))
            res.corr_mat = res.cov_mat / np.outer(d, d)
            
            # Intervalos de Confiança (95%)
            res.ci_p1 = 1.96 * d[0]
            if res.k_params > 1:
                res.ci_p2 = 1.96 * d[1]
                eigvals, _ = np.linalg.eigh(res.cov_mat)
                if np.any(eigvals < -1e-10):
                    res.cov_degenerada = True
        except Exception:
            res.cov_degenerada = True
            
        return res

@st.cache_data
def calcular_sse_matriz_exata(pwf_medidos, q_medidos, pe, p1_opt, p2_opt, is_fetkovich):
    res_malha = 75j 
    
    # Malha parametrizada em torno do ótimo para evitar cortes na região F-Test
    p1_min, p1_max = max(1e-8, p1_opt * 0.5), p1_opt * 1.5
    
    if is_fetkovich:
        p2_min, p2_max = max(0.5, p2_opt * 0.5), min(2.0, p2_opt * 1.5)
        j_grid, psat_grid = np.mgrid[p1_min:p1_max:res_malha, p2_min:p2_max:res_malha]
    else:
        p2_min, p2_max = max(14.7, p2_opt * 0.5), min(pe * 0.999, p2_opt * 1.5)
        j_grid, psat_grid = np.mgrid[p1_min:p1_max:res_malha, p2_min:p2_max:res_malha]
        
    j_grid = j_grid.astype(np.float32)
    psat_grid = psat_grid.astype(np.float32)
        
    pwf_brd = pwf_medidos[:, np.newaxis, np.newaxis].astype(np.float32)
    q_medidos_brd = q_medidos[:, np.newaxis, np.newaxis].astype(np.float32)
    
    if is_fetkovich:
        q_calc_tensor = ModelosIPR.fetkovich(pwf_brd, pe, j_grid, psat_grid)
    else:
        q_calc_tensor = ModelosIPR.hibrido_darcy_vogel(pwf_brd, pe, psat_grid, j_grid)
        
    sse_grid = np.sum((q_calc_tensor - q_medidos_brd)**2, axis=0)
            
    if is_fetkovich:
        q_min = ModelosIPR.fetkovich(pwf_medidos, pe, p1_opt, p2_opt)
    else:
        q_min = ModelosIPR.hibrido_darcy_vogel(pwf_medidos, pe, p2_opt, p1_opt)
        
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
    },
    "Caso 4: Gás/Turbulência (Preset Fetkovich)": {
        "Pe": 4000.0, "Pwf": [3500.0, 3000.0, 2000.0, 1000.0], "Q": [2000.0, 3500.0, 5800.0, 7200.0]
    }
}

if "ghost_curves" not in st.session_state:
    st.session_state["ghost_curves"] = []
st.session_state["ghost_curves"] = st.session_state["ghost_curves"][-5:]

st.set_page_config(page_title="Simulador IPR Científico", page_icon="🛢️", layout="wide")

if not salib_disponivel:
    st.error("⚠️ Biblioteca SALib não encontrada. O gráfico de Sobol falhará. Execute no terminal: pip install SALib")

st.title("🛢️ Simulador IPR - Forward Thermal Screening")
st.markdown("Framework de Pesquisa: Otimização OLS, Diagnóstico de Identificabilidade, Análise de Resíduos e Incertezas Termodinâmicas (Sobol).")

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
    param2_guess = st.sidebar.number_input("Chute n (Expoente)", value=0.8, min_value=0.2, max_value=2.0)
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
st.sidebar.header("🌡️ First-Order Mobility Approx")
ativar_termico = st.sidebar.checkbox("Ativar Acoplamento Forward", value=True)

correlacao_pvt = st.sidebar.selectbox(
    "Correlação PVT para Psat(T)", 
    ["Standing", "Vazquez-Beggs", "Glaso", "Petrosky-Farshad"]
)

t_ref = st.sidebar.number_input("T Ref PVT (°C)", value=25.0)
t_res = st.sidebar.number_input("T Reservatório (°C)", value=60.0)
ea_r = st.sidebar.slider("Constante Aparente (Ea/R) em K", 500.0, 5000.0, 2000.0, step=100.0)

st.sidebar.markdown("---")
st.sidebar.header("🌪️ Sensibilidade Estocástica (Sobol)")

qoi_sobol = st.sidebar.selectbox(
    "Quantity of Interest (QoI)",
    ["AOF (Potencial Máximo)", "Produtividade Média (Integral)", "Vazão Operacional Específica"]
)
pwf_op = 0.0
if qoi_sobol == "Vazão Operacional Específica":
    pwf_op = st.sidebar.number_input("Pwf Operacional (psi)", min_value=0.0, max_value=float(pe_campo), value=pe_campo/2)

var_sobol_pct = st.sidebar.slider("Incerteza Paramétrica (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
n_sobol = st.sidebar.selectbox("Amostras do Saltelli (N)", [512, 1024, 2048, 4096], index=1)

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
        with st.spinner("Resolvendo Sistema Inverso e Diagnóstico de Resíduos..."):
            try:
                # --- 1. HISTORY MATCHING TRF ---
                hm_service = HistoryMatchingService()
                res_calibracao = hm_service.calibrar(
                    well_name, pwf_campo, q_campo, pe_campo, 
                    param1_guess, param2_guess, param2_conhecido, is_fetkovich
                )

                if getattr(res_calibracao, 'psat_auto_locked', False):
                    st.warning("🔒 **Trava de Pseudo-Identificabilidade Ativada:** O Drawdown relativo medido é inferior a 10% (Escoamento puramente monofásico). O modelo fixou a saturação analiticamente para evitar colapso da Jacobiana e falsas correlações paramétricas.")

                st.subheader("Otimização OLS e Análise de Resíduos")
                col1, col2, col3, col4 = st.columns(4)
                
                if is_fetkovich:
                    col1.metric("C Base", f"{res_calibracao.param1_calibrado:.6e}")
                    col2.metric("n Otimizado", f"{res_calibracao.param2_calibrado:.3f}")
                else:
                    col1.metric("J_eff Base", f"{res_calibracao.param1_calibrado:.4f}")
                    col2.metric("Psat Otimizada", f"{res_calibracao.param2_calibrado:.1f} psi")
                
                col3.metric("RMSE Residual", f"{res_calibracao.rmse * fator_conv:.2f} {unidade_vazao}")
                
                if np.isnan(res_calibracao.r2_adj):
                    st.warning("⚠️ R² Ajustado indefinido: Faltam graus de liberdade (dados insuficientes).")
                elif res_calibracao.r2_adj < 0:
                    st.warning("⚠️ R² Ajustado negativo: o modelo penalizado ajusta pior que a média dos dados.")
                else:
                    col4.metric("R² Ajustado", f"{res_calibracao.r2_adj:.4f}")
                
                c_aic1, c_aic2, c_aic3 = st.columns(3)
                c_aic1.metric("WMAPE", f"{res_calibracao.wmape:.2f}%")
                c_aic2.metric("AIC (Akaike)", f"{res_calibracao.aic:.2f}")
                val_aicc = f"{res_calibracao.aicc:.2f}" if not np.isnan(res_calibracao.aicc) else "N/A"
                c_aic3.metric("AICc (Corrigido)", val_aicc)

                # Análise Gráfica de Resíduos
                fig_res, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(12, 4))
                ax_res1.axhline(0, color='red', linestyle='--', linewidth=1)
                ax_res1.scatter(pwf_campo, res_calibracao.residuos * fator_conv, color='blue', alpha=0.7)
                ax_res1.set_title("Resíduos vs Pressão (Heterocedasticidade)")
                ax_res1.set_xlabel("Pwf (psi)")
                ax_res1.set_ylabel(f"Erro ({unidade_vazao})")
                ax_res1.grid(True, linestyle=':')
                
                ax_res2.hist(res_calibracao.residuos * fator_conv, bins=10, color='green', alpha=0.6, edgecolor='black')
                ax_res2.set_title("Distribuição do Erro")
                ax_res2.set_xlabel(f"Resíduo ({unidade_vazao})")
                ax_res2.grid(True, linestyle=':')
                st.pyplot(fig_res)

                # --- 2. CURVA IPR E FORWARD TÉRMICO FÍSICO CORRIGIDO ---
                pwf_arr = np.linspace(pe_campo, 0, 100)
                
                if is_fetkovich:
                    q_arr_base = ModelosIPR.fetkovich(pwf_arr, pe_campo, res_calibracao.param1_calibrado, res_calibracao.param2_calibrado)
                    aof_base = ModelosIPR.fetkovich(0.0, pe_campo, res_calibracao.param1_calibrado, res_calibracao.param2_calibrado)
                else:
                    q_arr_base = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, res_calibracao.param2_calibrado, res_calibracao.param1_calibrado)
                    aof_base = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, res_calibracao.param2_calibrado, res_calibracao.param1_calibrado)

                if ativar_termico:
                    j_termico = CorretorTermico.ajustar_indice_J(res_calibracao.param1_calibrado, t_res, t_ref, ea_r)
                    psat_termica = CorretorTermico.ajustar_Psat(res_calibracao.param2_calibrado, t_res, t_ref, correlacao_pvt) if not is_fetkovich else res_calibracao.param2_calibrado
                    
                    if is_fetkovich:
                        q_arr_termico = ModelosIPR.fetkovich(pwf_arr, pe_campo, j_termico, psat_termica)
                        aof_termico = ModelosIPR.fetkovich(0.0, pe_campo, j_termico, psat_termica)
                    else:
                        q_arr_termico = ModelosIPR.hibrido_darcy_vogel(pwf_arr, pe_campo, psat_termica, j_termico)
                        aof_termico = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, psat_termica, j_termico)
                else:
                    aof_termico = aof_base 
                    psat_termica = res_calibracao.param2_calibrado

                q_arr_plot = q_arr_base * fator_conv
                aof_plot = aof_base * fator_conv
                
                aof_termico_plot = aof_termico * fator_conv
                q_arr_plot_t = q_arr_termico * fator_conv if ativar_termico else q_arr_plot
                
                st.session_state["ghost_curves"].append({"name": f"{well_name}", "q": q_arr_plot, "pwf": pwf_arr})

                fig_ipr, ax = plt.subplots(figsize=(10, 5))
                for ghost in st.session_state["ghost_curves"][:-1]:
                    ax.plot(ghost["q"], ghost["pwf"], color='gray', alpha=0.3, linestyle='--')
                
                ax.plot(q_arr_plot, pwf_arr, 'b-', linewidth=3, label=f'IPR Base OLS (AOF: {aof_plot:.0f})')
                if ativar_termico:
                    ax.plot(q_arr_plot_t, pwf_arr, color='#e53e3e', linewidth=3, linestyle='--', label=f'IPR Predição Térmica Semi-Acoplada (AOF: {aof_termico_plot:.0f})')
                ax.scatter(q_campo * fator_conv, pwf_campo, color='black', zorder=5, label='Dados Experimentais')
                ax.set_title("Curvas de Desempenho e Forward Thermal Screening (First-Order Approx)")
                ax.set_xlabel(f"Vazão de Produção ({unidade_vazao})")
                ax.set_ylabel("Pwf Dinâmica (psi)")
                ax.set_ylim(0, pe_campo + 500)
                limite_x = max(aof_termico_plot, aof_plot) * 1.1
                ax.set_xlim(0, limite_x)
                ax.grid(True, linestyle=':')
                ax.legend()
                st.pyplot(fig_ipr)

                # --- 3. IDENTIFICABILIDADE LOCAL E GLOBAL ---
                st.markdown("---")
                st.subheader("🔍 Diagnóstico de Identificabilidade Estrutural")
                
                if res_calibracao.cov_mat is not None and res_calibracao.k_params == 2:
                    
                    if not np.isnan(res_calibracao.cond_number):
                        if res_calibracao.cond_number < 1e2:
                            cond_status = "Ótimo (Identificável)"
                            cond_color = "normal"
                        elif res_calibracao.cond_number < 1e4:
                            cond_status = "Aceitável"
                            cond_color = "normal"
                        elif res_calibracao.cond_number < 1e6:
                            cond_status = "Mal Condicionado"
                            cond_color = "inverse"
                        else:
                            cond_status = "Não Identificável"
                            cond_color = "inverse"
                    else:
                        cond_status = "Indefinido"
                        
                    st.caption("**Estatística Assintótica Local:** Matriz de Covariância $\\sigma^2(J^TJ)^{-1}$ derivada da Jacobiana e Intervalos de Confiança (95%).")
                    col_mat1, col_mat2, col_mat3 = st.columns([1.5, 1.5, 1])
                    
                    lbl_1 = "C" if is_fetkovich else "J_eff"
                    lbl_2 = "n" if is_fetkovich else "Psat"
                    
                    df_cov = pd.DataFrame(res_calibracao.cov_mat, columns=[lbl_1, lbl_2], index=[lbl_1, lbl_2])
                    df_cor = pd.DataFrame(res_calibracao.corr_mat, columns=[lbl_1, lbl_2], index=[lbl_1, lbl_2])
                    
                    with col_mat1:
                        st.write("Matriz de Covariância:")
                        st.dataframe(df_cov.style.format("{:.3e}"))
                    with col_mat2:
                        st.write(f"Intervalos de Confiança (95%):")
                        st.write(f"- **{lbl_1}:** {res_calibracao.param1_calibrado:.4e} ± {res_calibracao.ci_p1:.4e}")
                        st.write(f"- **{lbl_2}:** {res_calibracao.param2_calibrado:.2f} ± {res_calibracao.ci_p2:.2f}")
                    with col_mat3:
                        st.metric("Condition Number (κ)", f"{res_calibracao.cond_number:.1e}" if not np.isnan(res_calibracao.cond_number) else "NaN", delta=cond_status, delta_color=cond_color if 'cond_color' in locals() else "off")

                j_grid, psat_grid, sse_grid, sse_min = calcular_sse_matriz_exata(
                    pwf_campo, q_campo, pe_campo, res_calibracao.param1_calibrado, res_calibracao.param2_calibrado, is_fetkovich
                )

                N_dados = len(pwf_campo)
                if N_dados < 8:
                    st.warning("⚠️ **Aviso Metodológico:** A matriz de calibração possui menos de 8 pontos experimentais. Os graus de liberdade limitam a precisão estatística assintótica da região de confiança F-Test.")

                p_livres = res_calibracao.k_params
                v_df = max(1, N_dados - p_livres)
                
                f_68 = stats.f.ppf(0.68, dfn=p_livres, dfd=v_df)
                f_95 = stats.f.ppf(0.95, dfn=p_livres, dfd=v_df)
                f_99 = stats.f.ppf(0.99, dfn=p_livres, dfd=v_df)
                
                sse_limiar_68 = sse_min * (1.0 + (float(p_livres) / v_df) * f_68)
                sse_limiar_95 = sse_min * (1.0 + (float(p_livres) / v_df) * f_95)
                sse_limiar_99 = sse_min * (1.0 + (float(p_livres) / v_df) * f_99)
                    
                mask_valid = ~np.isnan(sse_grid)
                area_pixels = np.sum((sse_grid <= sse_limiar_95) & mask_valid)
                area_pct = (area_pixels / np.sum(mask_valid)) * 100 if np.sum(mask_valid) > 0 else 0.0

                col_diag1, col_diag2 = st.columns(2)
                col_diag1.metric(f"Região de Confiança Global (F-95%)", f"{area_pct:.1f}% do Domínio de Avaliação")
                col_diag2.metric("Mínimo Global OLS", f"{sse_min:.1f} (SSE)")

                label_x = 'C' if is_fetkovich else 'J_eff'
                label_y = 'n' if is_fetkovich else 'Psat'

                if res_calibracao.cov_mat is not None and res_calibracao.k_params == 2:
                    if getattr(res_calibracao, 'cov_degenerada', False):
                        st.error("🚨 **Matriz Jacobiana Degenerada/Hessiana Mal Condicionada detectada.** A elipse assintótica não pôde ser resolvida com estabilidade (Autovalores não-físicos).")
                    else:
                        eigvals, eigvecs = np.linalg.eigh(res_calibracao.cov_mat)
                        eigvals = np.maximum(eigvals, 0)
                        
                        chi2_val = stats.chi2.ppf(0.95, df=2)
                        t_ang = np.linspace(0, 2*np.pi, 100)
                        circle = np.vstack((np.cos(t_ang), np.sin(t_ang)))
                        transform = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals * chi2_val, 0)))
                        ellipse = transform @ circle
                        
                        x_ell = res_calibracao.param1_calibrado + ellipse[0, :]
                        y_ell = res_calibracao.param2_calibrado + ellipse[1, :]
                        
                        fig_2d = go.Figure()
                        fig_2d.add_trace(go.Contour(z=np.log10(sse_grid - sse_min + 1), x=j_grid[:,0], y=psat_grid[0,:], colorscale='Blues', opacity=0.6, name='Log(SSE) Global'))
                        fig_2d.add_trace(go.Scatter(x=x_ell, y=y_ell, mode='lines', line=dict(color='red', width=2), name='Elipse de Covariância ($\chi^2_{95\%}$)'))
                        fig_2d.add_trace(go.Scatter(x=[res_calibracao.param1_calibrado], y=[res_calibracao.param2_calibrado], mode='markers', marker=dict(color='red', size=8, symbol='x'), name='Ótimo Local'))
                        
                        fig_2d.update_layout(title="Projeção 2D (Identificabilidade Local vs Global)", xaxis_title=label_x, yaxis_title=label_y, height=450)
                        st.plotly_chart(fig_2d, use_container_width=True)

                fig_3d = go.Figure(data=[go.Surface(z=sse_grid, x=j_grid, y=psat_grid, colorscale='Viridis')])
                fig_3d.add_trace(go.Scatter3d(
                    x=[res_calibracao.param1_calibrado], y=[res_calibracao.param2_calibrado], z=[sse_min],
                    mode='markers', marker=dict(symbol='diamond', size=7, color='red'), name='Mínimo Estrito'
                ))
                fig_3d.update_layout(scene=dict(xaxis_title=label_x, yaxis_title=label_y, zaxis_title='SSE'), height=550)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.caption(f"**Nota Analítica (Limiares de Verossimilhança F-Test):** Corte 68% (SSE \u2264 {sse_limiar_68:.1f}) | Corte 95% (SSE \u2264 {sse_limiar_95:.1f}) | Corte 99% (SSE \u2264 {sse_limiar_99:.1f}).")

                S1 = [0.0, 0.0, 0.0]
                ST = [0.0, 0.0, 0.0]
                P_90 = P_50 = P_10 = 0.0
                qoi_label = ""

                # --- 4. SOBOL VETORIZADO COM SELEÇÃO DINÂMICA DE QoI ---
                if ativar_termico:
                    st.markdown("---")
                    st.subheader(f"🌪️ Risco Operacional e Sensibilidade Térmica Global (\u00B1{var_sobol_pct}%)")
                    
                    with st.spinner(f"Processando Matriz Tensorial de Sobol via Fast Saltelli (N={n_sobol})..."):
                        delta = var_sobol_pct / 100.0
                        
                        bounds_t_res = [max(0.1, t_res * (1.0 - delta)), t_res * (1.0 + delta)]
                        bounds_t_ref = [max(0.1, t_ref * (1.0 - delta)), t_ref * (1.0 + delta)]
                        bounds_ea_r  = [max(1.0, ea_r * (1.0 - delta)), ea_r * (1.0 + delta)]

                        problem = {
                            'num_vars': 3, 
                            'names': ['T_Reservatorio', 'T_Referencia', 'Ea/R'], 
                            'bounds': [bounds_t_res, bounds_t_ref, bounds_ea_r]
                        }
                        
                        param_values = sobol_sample.sample(problem, n_sobol)
                        st.info(f"Amostras efetivas da matriz Sobol-Saltelli: {len(param_values):,}")
                        
                        t_res_s = param_values[:, 0]
                        t_ref_s = param_values[:, 1]
                        ea_r_s  = param_values[:, 2]

                        j_pert_arr = CorretorTermico.ajustar_indice_J(res_calibracao.param1_calibrado, t_res_s, t_ref_s, ea_r_s)
                        
                        # Definição dinâmica da Variável de Saída (QoI) com base no UI
                        if qoi_sobol == "AOF (Potencial Máximo)":
                            qoi_label = "AOF"
                            if is_fetkovich:
                                psat_pert_arr = np.full_like(t_res_s, res_calibracao.param2_calibrado)
                                q_output_tensorial = ModelosIPR.fetkovich(0.0, pe_campo, j_pert_arr, psat_pert_arr) * fator_conv
                            else:
                                psat_pert_arr = CorretorTermico.ajustar_Psat(res_calibracao.param2_calibrado, t_res_s, t_ref_s, correlacao_pvt)
                                q_output_tensorial = ModelosIPR.hibrido_darcy_vogel(0.0, pe_campo, psat_pert_arr, j_pert_arr) * fator_conv
                                
                        elif qoi_sobol == "Vazão Operacional Específica":
                            qoi_label = f"q @ {pwf_op} psi"
                            if is_fetkovich:
                                psat_pert_arr = np.full_like(t_res_s, res_calibracao.param2_calibrado)
                                q_output_tensorial = ModelosIPR.fetkovich(pwf_op, pe_campo, j_pert_arr, psat_pert_arr) * fator_conv
                            else:
                                psat_pert_arr = CorretorTermico.ajustar_Psat(res_calibracao.param2_calibrado, t_res_s, t_ref_s, correlacao_pvt)
                                q_output_tensorial = ModelosIPR.hibrido_darcy_vogel(pwf_op, pe_campo, psat_pert_arr, j_pert_arr) * fator_conv
                                
                        else:
                            qoi_label = "Média Integral"
                            pwf_frac = np.linspace(0, 1, 50)
                            pwf_array_fixo = pe_campo * pwf_frac
                            pwf_brd = pwf_array_fixo[np.newaxis, :] 
                            j_brd = j_pert_arr[:, np.newaxis]       
                            
                            if is_fetkovich:
                                psat_brd = np.full_like(j_brd, res_calibracao.param2_calibrado)
                                q_tensor = ModelosIPR.fetkovich(pwf_brd, pe_campo, j_brd, psat_brd)
                            else:
                                psat_pert_arr = CorretorTermico.ajustar_Psat(res_calibracao.param2_calibrado, t_res_s, t_ref_s, correlacao_pvt)
                                psat_brd = psat_pert_arr[:, np.newaxis]
                                q_tensor = ModelosIPR.hibrido_darcy_vogel(pwf_brd, pe_campo, psat_brd, j_brd)
                                
                            try:
                                integral_q = np.trapezoid(q_tensor, pwf_array_fixo, axis=1)
                            except AttributeError:
                                integral_q = np.trapz(q_tensor, pwf_array_fixo, axis=1)
                                
                            q_output_tensorial = (integral_q / pe_campo) * fator_conv
                            
                        # Trava de Segurança Estatística do Sobol
                        mask_valid = np.isfinite(q_output_tensorial)
                        
                        if not np.all(mask_valid):
                            st.error("🚨 Valores numéricos inválidos detectados na avaliação do QoI. Análise de Sobol abortada.")
                        elif np.var(q_output_tensorial) < 1e-12:
                            st.warning("⚠️ **Alerta Estatístico:** A variância do QoI selecionado é virtualmente nula. Índices de Sobol apresentarão instabilidade numérica severa.")
                        else:
                            P_90 = np.percentile(q_output_tensorial, 10)
                            P_50 = np.percentile(q_output_tensorial, 50)
                            P_10 = np.percentile(q_output_tensorial, 90)

                            c_p10, c_p50, c_p90 = st.columns(3)
                            c_p90.metric(f"P90 (Pessimista)", f"{P_90:.0f} {unidade_vazao}")
                            c_p50.metric(f"P50 (Mediana)", f"{P_50:.0f} {unidade_vazao}")
                            c_p10.metric(f"P10 (Otimista)", f"{P_10:.0f} {unidade_vazao}")
                            
                            try:
                                Si = sobol.analyze(problem, q_output_tensorial)
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
                                    title=f"Decomposição de Variância (QoI: {qoi_label})",
                                    xaxis_title="Índice de Sobol (Fração da Incerteza Explicada)",
                                    barmode='group', height=400
                                )
                                st.plotly_chart(fig_sobol, use_container_width=True)
                                
                                idx_max_s1 = np.argmax(S1)
                                nome_max = problem['names'][idx_max_s1]
                                
                                st.caption(f"**Análise da Dissertação (Vetorização Tensorial):** O *Quantity of Interest (QoI)* adotado foi **{qoi_label}**. Avaliadas {len(q_output_tensorial)} aproximações semi-acopladas paramétricas. O indicador atesta que **{nome_max}** rege isoladamente **{S1[idx_max_s1]*100:.1f}%** da incerteza propagada na entrega de produção.")
                                st.info("""Nota Metodológica (First-Order Mobility Approx): A variação térmica deforma o Índice Efetivo ($J_{eff}$) por via da viscosidade combinada à expansão de $P_{sat}$ (Correlata PVT), enquanto $k_{ro}$, $B_o$, $R_s$ e $P_e$ foram modelados como estacionários.""")
                            except Exception as e:
                                st.error(f"Erro Crítico na Análise de Sobol: Falha na extração de variância pelo método de Saltelli. Detalhes: {e}")
                else:
                    st.info("Ative o 'Acoplamento Forward' na barra lateral para habilitar a Análise Estocástica de Risco P10/P50/P90.")

                # --- EXPORTAÇÃO BLINDADA EM LATEX ---
                st.markdown("---")
                st.subheader("📥 Geração de Relatório Físico-Estatístico")
                
                str_j = f"{res_calibracao.param1_calibrado:.4f}"
                str_p = f"{res_calibracao.param2_calibrado:.2f}"
                str_pe = f"{pe_campo:.2f}"
                str_rmse = f"{res_calibracao.rmse * fator_conv:.2f}"
                str_r2 = f"{res_calibracao.r2_adj:.4f}" if not np.isnan(res_calibracao.r2_adj) else "N/A"
                str_aic = f"{res_calibracao.aic:.2f}"
                str_wmape = f"{res_calibracao.wmape:.2f}"
                str_aof = f"{aof_plot:.2f}"
                
                str_cov = "Matriz de Covari\\^ancia N\\~ao Calculada (Problema Degenerado ou Satura\\c{{c}}\\~ao Fixada)"
                if res_calibracao.cov_mat is not None and res_calibracao.k_params == 2 and not getattr(res_calibracao, 'cov_degenerada', False):
                    str_cov = (f"C(1,1)={res_calibracao.cov_mat[0,0]:.2e}, C(2,2)={res_calibracao.cov_mat[1,1]:.2e}, "
                               f"C(1,2)={res_calibracao.cov_mat[0,1]:.2e} ($\\kappa$={res_calibracao.cond_number:.1e})")
                
                tex_base = f"""\\documentclass{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\begin{{document}}

\\section*{{Memorial F\\'isico-Estat\\'istico - Po\\c{{c}}o: {well_name}}}

\\subsection*{{1. Sele\\c{{c}}\\~ao de Modelo e Problema Inverso (OLS)}}
\\begin{{itemize}}
  \\item R2 Ajustado: {str_r2}
  \\item RMSE Residual do Truncamento: {str_rmse} {unidade_vazao}
  \\item WMAPE: {str_wmape}\\%
  \\item Crit\\'erio de Informa\\c{{c}}\\~ao (AIC): {str_aic}
  \\item Parametro 1 Convergido: {str_j}
  \\item Parametro 2 Convergido: {str_p}
  \\item Identificabilidade Estrita: {str_cov}
\\end{{itemize}}

"""

                if ativar_termico:
                    str_aof_t = f"{(aof_termico * fator_conv):.2f}"
                    str_mult = f"{(CorretorTermico.calcular_razao_viscosidade(t_res, t_ref, ea_r)):.4f}"
                    str_psat_t = f"{psat_termica:.2f}"
                    str_delta = f"{((aof_termico - aof_base) * fator_conv):+.2f}"
                    tex_qoi = qoi_label.replace("@", "\\@")
                    
                    tex_termico = f"""\\subsection*{{2. Forward Thermal Screening (First-Order Approx)}}
\\begin{{itemize}}
  \\item Delta de Temperatura: {t_ref}C a {t_res}C
  \\item Raz\\~ao Aparente Constante F\\'isica (Ea/R): {ea_r} K
  \\item Correla\\c{{c}}\\~ao PVT Acoplada (Psat): {correlacao_pvt}
  \\item Multiplicador de Mobilidade (Viscosidade): {str_mult}
  \\item Psat Corrigida Termicamente: {str_psat_t} psi
  \\item AOF Base (Frio): {str_aof} {unidade_vazao}
  \\item AOF T\\'ermico (Aquecido): {str_aof_t} {unidade_vazao}
  \\item Delta AOF Absoluto: {str_delta} {unidade_vazao}
\\end{{itemize}}

\\subsection*{{3. An\\'alise de Risco Estoc\\'astica e Sobol (Janela $\\pm{var_sobol_pct}\\%, QoI: {tex_qoi}$)}}
\\begin{{itemize}}
  \\item P90 (Produtividade Conservadora): {P_90:.0f} {unidade_vazao}
  \\item P50 (Produtividade Mediana): {P_50:.0f} {unidade_vazao}
  \\item P10 (Produtividade Otimista): {P_10:.0f} {unidade_vazao}
  \\item $S_1$ (Temp. Reservatorio): {S1[0]:.4f} | $S_T$: {ST[0]:.4f}
  \\item $S_1$ (Temp. Referencia): {S1[1]:.4f} | $S_T$: {ST[1]:.4f}
  \\item $S_1$ (Razao Ea/R): {S1[2]:.4f} | $S_T$: {ST[2]:.4f}
\\end{{itemize}}
"""
                else:
                    tex_termico = f"""\\subsection*{{2. Potencial M\\'aximo OLS}}
AOF estabilizado: {str_aof} {unidade_vazao}.

"""

                tex_sobol = "" 
                
                latex_content = tex_base + tex_termico + tex_sobol + "\\end{document}\n"

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(label="🖩 Baixar Memorial LaTeX", data=latex_content, file_name=f"memorial_{well_name}.tex")
                with col_btn2:
                    df_rel = pd.DataFrame({
                        "Parâmetro": ["Poço", "RMSE", "R2 Adj", "AIC", "WMAPE", "AOF Base", "AOF Térmico", "P90 (QoI)", "P50 (QoI)", "P10 (QoI)"],
                        "Valor": [well_name, str_rmse, str_r2, str_aic, str_wmape, str_aof, 
                                  f"{(aof_termico * fator_conv):.1f}" if ativar_termico else "-",
                                  f"{P_90:.1f}" if ativar_termico else "-", f"{P_50:.1f}" if ativar_termico else "-", f"{P_10:.1f}" if ativar_termico else "-"]
                    })
                    st.download_button(label="📄 Baixar Tensor Numérico CSV", data=df_rel.to_csv(index=False, sep=';').encode('utf-8-sig'), file_name=f"dados_{well_name}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Erro Crítico Matemático/Estatístico: {e}")