import logging
import numpy as np
from typing import Tuple
from reservatorio.domain.models import PocoFisico
from reservatorio.domain.ipr_models import IPRStrategy

logger = logging.getLogger(__name__)

class AnaliseProdutividadeService:
    """Serviço de aplicação para orquestrar a geração da curva IPR e validar as regras físicas."""
    
    def __init__(self, modelo_ipr: IPRStrategy):
        self.modelo_ipr = modelo_ipr
        
    def gerar_perfil_inflow(self, dados_poco: dict, J_calibrado: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
        try:
            poco = PocoFisico(**dados_poco)
            q_arr, pwf_arr, _, aof = self.modelo_ipr.calcular_curva(poco, J_in=J_calibrado)
            
            logger.info(f"[{poco.nome}] AOF Calculada: {aof:.1f} STB/d")
            return q_arr, pwf_arr, aof
            
        except ValueError as e:
            logger.error(f"Erro de validação física: {e}")
            raise