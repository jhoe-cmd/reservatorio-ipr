import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def calcular_skin(ko: float, kd: float, rd: float, rw: float) -> float:
    """Calcula o Fator de Dano de Hawkins equivalente."""
    for val in [ko, kd, rd, rw]:
        if not np.isfinite(val):
            raise ValueError("Valores NaN ou Inf não são permitidos.")
            
    if ko <= 0 or kd <= 0:
        raise ValueError("Permeabilidades devem ser estritamente positivas.")
    if rw <= 0:
        raise ValueError("O raio do poço (rw) deve ser estritamente positivo.")
    if rd <= rw:
        raise ValueError("O raio da zona alterada (rd) deve ser maior que rw.")
        
    S = (ko / kd - 1) * np.log(rd / rw)
    
    if S < -5 or S > 20:
        logger.warning(f"Fator de Dano (S = {S:.2f}) fora dos limites heurísticos (-5 a 20).")
        
    return float(S)