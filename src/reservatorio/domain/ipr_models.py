import numpy as np

class ModelosIPR:
    """
    Camada de Domínio: Contém as leis físicas de escoamento.
    Nenhuma dependência de interface gráfica (Streamlit) deve entrar aqui.
    """
    @staticmethod
    def fetkovich(pwf: np.ndarray, pe: float, c: float, n: float) -> np.ndarray:
        """
        Calcula a vazão teórica pelo modelo de Fetkovich para gás/alta turbulência.
        q = C * (Pe^2 - Pwf^2)^n
        """
        # Garante que a Pwf não ultrapassa a Pe por ruído numérico
        pwf_clipped = np.clip(pwf, 0, pe)
        
        # Cálculo vetorizado da equação de Fetkovich
        return c * (pe**2 - pwf_clipped**2)**n
    
    @staticmethod
    def darcy_linear(pwf, pe, j):
        """Modelo linear para regime monofásico."""
        # Garante que não haja vazão negativa se Pwf > Pe
        drawdown = np.maximum(0, pe - pwf)
        return j * drawdown

    @staticmethod
    def vogel_classico(pwf, pe, qmax):
        """Modelo de Vogel para reservatórios saturados (Pe <= Psat)."""
        ratio = pwf / pe
        # Evita vazão negativa para Pwf > Pe
        ratio = np.clip(ratio, 0, 1)
        return qmax * (1 - 0.2 * ratio - 0.8 * (ratio**2))

    @staticmethod
    def hibrido_darcy_vogel(pwf, pe, psat, j):
        """Modelo composto: Darcy acima da Psat, Vogel abaixo da Psat."""
        pwf = np.atleast_1d(pwf) # Garante que funcione com arrays ou escalares
        q = np.zeros_like(pwf, dtype=float)
        
        # Vazão no ponto de saturação
        q_sat = j * (pe - psat)
        # Qmax projetado pela regra de Vogel modificada
        qmax_projetado = q_sat + (j * psat) / 1.8
        
        for i, p in enumerate(pwf):
            if p >= pe:
                q[i] = 0.0
            elif p >= psat:
                # Regime Monofásico (Darcy)
                q[i] = j * (pe - p)
            else:
                # Regime Bifásico (Vogel)
                ratio = p / psat
                q[i] = q_sat + (qmax_projetado - q_sat) * (1 - 0.2 * ratio - 0.8 * (ratio**2))
                
        return q if len(q) > 1 else q[0]

    @staticmethod
    def fetkovich(pwf, pe, qmax, n):
        """Modelo empírico de Fetkovich. 'n' varia de 0.5 a 1.0."""
        ratio = pwf / pe
        ratio = np.clip(ratio, 0, 1)
        return qmax * (1 - (ratio**2))**n