from abc import ABC, abstractmethod
import numpy as np
from typing import List
from reservatorio.config import ReservoirConstants

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
        J, Psat = params[0], params[1]
        q_teorico = np.empty_like(pwf_medidos)
        
        mask_darcy = pwf_medidos >= Psat
        mask_vogel = ~mask_darcy
        
        q_teorico[mask_darcy] = J * (Pe - pwf_medidos[mask_darcy])
        
        qb = J * (Pe - Psat)
        fator_vogel = (1.0 - 0.2*(pwf_medidos[mask_vogel]/Psat) - 0.8*(pwf_medidos[mask_vogel]/Psat)**2) / ReservoirConstants.VOGEL_CONSTANT
        q_teorico[mask_vogel] = qb + (J * Psat) * fator_vogel
        
        return q_teorico - q_medidos