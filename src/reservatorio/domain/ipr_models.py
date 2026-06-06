from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from .models import PocoFisico
from reservatorio.config import ReservoirConstants

ArrayLike = Union[float, np.floating, npt.NDArray[np.float64]]

def fator_vogel_math(pwf: ArrayLike, psat: float) -> ArrayLike:
    return (1.0 - 0.2 * (pwf / psat) - 0.8 * (pwf / psat)**2) / ReservoirConstants.VOGEL_CONSTANT

class IPRStrategy(ABC):
    @abstractmethod
    def calcular_curva(self, poco: PocoFisico, J_in: float = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
        pass

class DarcyVogelHibridoIPR(IPRStrategy):
    def calcular_curva(self, poco: PocoFisico, J_in: float = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
        J_ipr = J_in if J_in is not None else (
            poco.q_test / (poco.Pe - poco.Pwf_test) if poco.Pwf_test >= poco.Psat 
            else poco.q_test / ((poco.Pe - poco.Psat) + poco.Psat * fator_vogel_math(poco.Pwf_test, poco.Psat))
        )

        qb = J_ipr * (poco.Pe - poco.Psat)
        qmax = qb + (J_ipr * poco.Psat) / ReservoirConstants.VOGEL_CONSTANT
        
        pwf_arr = np.linspace(0, poco.Pe, ReservoirConstants.DEFAULT_POINTS, dtype=np.float64)
        q_arr = np.zeros_like(pwf_arr)
        
        mask_darcy = pwf_arr >= poco.Psat
        mask_vogel = ~mask_darcy
        
        q_arr[mask_darcy] = J_ipr * (poco.Pe - pwf_arr[mask_darcy])
        q_arr[mask_vogel] = qb + (J_ipr * poco.Psat) * fator_vogel_math(pwf_arr[mask_vogel], poco.Psat)
        
        return q_arr, pwf_arr, J_ipr, qmax