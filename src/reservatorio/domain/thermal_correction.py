import numpy as np

class CorretorTermico:
    """
    Módulo para acoplamento térmico da dissertação.
    Calcula a variação do Índice de Produtividade (J) em função da 
    viscosidade e do campo de temperatura.
    """
    @staticmethod
    def fator_viscosidade(t_res: float, t_ref: float, b_constante: float = 0.025) -> float:
        """
        Calcula a razão mu_ref / mu_atual com base na lei exponencial de temperatura.
        Quanto mais quente, menor a viscosidade, maior o fluxo.
        """
        # mu(T) = mu_ref * exp(-b * (T - T_ref))
        # Retorna o multiplicador: exp(b * (T - T_ref))
        return np.exp(b_constante * (t_res - t_ref))

    @staticmethod
    def ajustar_indice_J(j_base: float, t_res: float, t_ref: float, incerteza_pct: float = 0.0) -> float:
        """
        Ajusta o J (Índice de Produtividade) aplicando o campo de temperatura
        e a variação estocástica (+- 5%) da dissertação.
        """
        fator_t = CorretorTermico.fator_viscosidade(t_res, t_ref)
        j_termico = j_base * fator_t
        
        # Aplica a perturbação da análise de sensibilidade
        j_perturbado = j_termico * (1.0 + (incerteza_pct / 100.0))
        
        return j_perturbado