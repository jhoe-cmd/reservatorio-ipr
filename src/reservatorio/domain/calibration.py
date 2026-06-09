from abc import ABC, abstractmethod
import numpy as np
from typing import List

# Importação correta do nosso motor matemático
from reservatorio.domain.ipr_models import ModelosIPR

class CalibrationStrategy(ABC):
    @abstractmethod
    def residuals(self, params: List[float], pwf_medidos: np.ndarray, q_medidos: np.ndarray, Pe: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

class DarcyVogelCalibration(CalibrationStrategy):
    def get_model_name(self) -> str:
        return "Darcy-Vogel Híbrido (Multi-parâmetro)"

    def residuals(self, params: List[float], pwf_medidos: np.ndarray, q_medidos: np.ndarray, Pe: float) -> np.ndarray:
        # 1. Desempacota os parâmetros que o otimizador (TRF) está testando nesta iteração
        J, Psat = params[0], params[1]
        
        # 2. Chama a Camada de Domínio para fazer todo o cálculo pesado instantaneamente
        q_teorico = ModelosIPR.hibrido_darcy_vogel(
            pwf=pwf_medidos, 
            pe=Pe, 
            psat=Psat, 
            j=J
        )
        
        # 3. Retorna o resíduo (Erro) para o otimizador
        return q_teorico - q_medidos