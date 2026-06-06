import logging
import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from reservatorio.domain.calibration import CalibrationStrategy
from reservatorio.domain.models import CalibrationResult
from reservatorio.infrastructure.repositories import CalibrationRepository

logger = logging.getLogger(__name__)

class HistoryMatchingService:
    def __init__(self, strategy: CalibrationStrategy, repository: CalibrationRepository):
        self.strategy = strategy
        self.repository = repository

    def calibrar(self, well_name: str, pwf_medidos: np.ndarray, q_medidos: np.ndarray, Pe: float, J_guess: float, Psat_guess: float) -> CalibrationResult:
        if len(pwf_medidos) < 3:
            raise ValueError("Calibração multiparâmetro exige pelo menos 3 pontos de teste.")

        resultado = least_squares(
            self.strategy.residuals,
            x0=[J_guess, Psat_guess],
            args=(pwf_medidos, q_medidos, Pe),
            bounds=([1e-5, 1e-5], [np.inf, Pe]),
            method='trf'
        )
        
        J_opt, Psat_opt = resultado.x
        residuos_finais = resultado.fun
        q_preditos = q_medidos + residuos_finais
        
        rmse = np.sqrt(np.mean(residuos_finais**2))
        
        # Prevenção contra divisão por zero no MAPE
        mask_nonzero = q_medidos > 1e-5
        mape = mean_absolute_percentage_error(q_medidos[mask_nonzero], q_preditos[mask_nonzero]) if np.any(mask_nonzero) else 0.0

        result = CalibrationResult(
            well_name=well_name,
            model=self.strategy.get_model_name(),
            J_calibrado=J_opt,
            Psat_calibrado=Psat_opt,
            rmse=rmse,
            mae=mean_absolute_error(q_medidos, q_preditos),
            mape=mape,
            r2=r2_score(q_medidos, q_preditos),
            bias=np.mean(residuos_finais),
            success=resultado.success,
            nfev=resultado.nfev,
            cost=resultado.cost,
            message=resultado.message
        )
        
        self.repository.save(result)
        logger.info(f"[{well_name}] Calibração salva. J: {J_opt:.3f} | RMSE: {rmse:.2f}")
        return result