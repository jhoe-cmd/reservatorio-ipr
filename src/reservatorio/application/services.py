import logging
import numpy as np
from typing import Tuple
from reservatorio.domain.models import PocoFisico
from reservatorio.domain.ipr_models import ModelosIPR

logger = logging.getLogger(__name__)

class AnaliseProdutividadeService:
    """Serviço de aplicação para orquestrar a geração da curva IPR e validar as regras físicas."""
    
    # Removemos o __init__ antigo que exigia a classe IPRStrategy deletada.
        
    def gerar_perfil_inflow(self, dados_poco: dict, J_calibrado: float) -> Tuple[np.ndarray, np.ndarray, float]:
        try:
            # 1. Valida as regras físicas instanciando a entidade de Domínio
            poco = PocoFisico(**dados_poco)
            
            # 2. Cria o array de pressões (de Pe até 0)
            pwf_arr = np.linspace(poco.Pe, 0, 50)
            
            # 3. Usa a nossa nova Camada de Domínio para calcular o array de vazões
            q_arr = ModelosIPR.hibrido_darcy_vogel(
                pwf=pwf_arr, 
                pe=poco.Pe, 
                psat=poco.Psat, 
                j=J_calibrado
            )
            
            # 4. Calcula o AOF passando Pwf = 0
            aof = ModelosIPR.hibrido_darcy_vogel(
                pwf=np.array([0.0]), 
                pe=poco.Pe, 
                psat=poco.Psat, 
                j=J_calibrado
            )[0]
            
            logger.info(f"[{poco.nome}] AOF Calculada: {aof:.1f} STB/d")
            return q_arr, pwf_arr, aof
            
        except ValueError as e:
            logger.error(f"Erro de validação física: {e}")
            raise