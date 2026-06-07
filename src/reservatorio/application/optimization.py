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
    #inserção de trecho de codigo:
    import numpy as np
import warnings

def generate_rmse_surface(pwf_medidos, q_medidos, Pe, J_opt, Psat_opt, n_j=50, n_psat=50, delta_psi=5.0):
    j_min = max(0.1, J_opt * 0.5)
    j_max = min(10.0, J_opt * 1.5)
    psat_min = max(1000.0, Psat_opt * 0.7)
    psat_max = min(Pe * 0.999, Psat_opt * 1.2)

    J_vals = np.linspace(j_min, j_max, n_j)
    Psat_vals = np.linspace(psat_min, psat_max, n_psat)
    J_grid, Psat_grid = np.meshgrid(J_vals, Psat_vals)
    
    MSE_grid = np.zeros_like(J_grid)
    invalid_mask = np.zeros_like(J_grid, dtype=bool)

    for pwf_real, q_real in zip(pwf_medidos, q_medidos):
        q_sat_grid = J_grid * (Pe - Psat_grid)
        pwf_calc = np.full_like(J_grid, np.nan)
        
        # --- Condição Monofásica (Darcy) ---
        mask_darcy = q_real <= q_sat_grid
        pwf_calc[mask_darcy] = Pe - (q_real / J_grid[mask_darcy])
        
        # --- Condição Bifásica (Vogel) ---
        mask_vogel = ~mask_darcy
        q_max_vogel = (J_grid[mask_vogel] * Psat_grid[mask_vogel]) / 1.8
        
        a = 0.8
        b = 0.2
        c = ((q_real - q_sat_grid[mask_vogel]) / q_max_vogel) - 1.0
        
        delta = b**2 - 4*a*c
        
        invalid_mask[mask_vogel] |= (delta < 0)
        
        mask_valid_delta = delta >= 0
        delta_valid = delta[mask_valid_delta]
        
        x1 = (-b + np.sqrt(delta_valid)) / (2 * a)
        x2 = (-b - np.sqrt(delta_valid)) / (2 * a)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Calcula as raízes apenas para o array encolhido (válido)
            x_valid_partial = np.where((x1 >= 0) & (x1 <= 1), x1,
                               np.where((x2 >= 0) & (x2 <= 1), x2, np.nan))
        
        # --- CORREÇÃO DO BROADCASTING AQUI ---
        # Remonta o array para o tamanho original do mask_vogel usando NaN como preenchimento
        x_valid_full = np.full_like(delta, np.nan)
        x_valid_full[mask_valid_delta] = x_valid_partial
        
        # Agora os tamanhos são compatíveis (2247,) com (2247,)
        invalid_mask[mask_vogel] |= np.isnan(x_valid_full)
        
        pwf_vogel_valid = np.full_like(delta, np.nan)
        pwf_vogel_valid[mask_valid_delta] = Psat_grid[mask_vogel][mask_valid_delta] * x_valid_partial
        pwf_calc[mask_vogel] = pwf_vogel_valid
        
        MSE_grid[~invalid_mask] += (pwf_real - pwf_calc[~invalid_mask])**2

    RMSE_grid = np.sqrt(MSE_grid / len(q_medidos))
    RMSE_grid[invalid_mask] = np.nan

    rmse_min = np.nanmin(RMSE_grid)
    limiar_incerteza = rmse_min + delta_psi

    mask_valid_domain = ~np.isnan(RMSE_grid)
    mask_incerteza = (RMSE_grid <= limiar_incerteza) & mask_valid_domain
    area_incerteza_pct = (np.sum(mask_incerteza) / np.sum(mask_valid_domain)) * 100

    j_incerteza = J_grid[mask_incerteza]
    psat_incerteza = Psat_grid[mask_incerteza]
    
    condicionamento_ci = np.nan
    if len(j_incerteza) > 2:
        j_norm = (j_incerteza - np.mean(j_incerteza)) / (np.std(j_incerteza) + 1e-8)
        psat_norm = (psat_incerteza - np.mean(psat_incerteza)) / (np.std(psat_incerteza) + 1e-8)
        
        cov_matrix = np.cov(j_norm, psat_norm)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        
        if len(eigenvalues) == 2:
            lambda_max = np.max(eigenvalues)
            lambda_min = np.min(eigenvalues)
            if lambda_min > 1e-8:
                condicionamento_ci = lambda_max / lambda_min

    return {
        "J_grid": J_grid,
        "Psat_grid": Psat_grid,
        "RMSE_grid": RMSE_grid,
        "rmse_min": rmse_min,
        "limiar_incerteza": limiar_incerteza,
        "area_incerteza_pct": area_incerteza_pct,
        "condicionamento_ci": condicionamento_ci
    }