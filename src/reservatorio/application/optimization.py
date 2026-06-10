import logging
import numpy as np
import warnings
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

    def calibrar(self, well_name: str, pwf_medidos: np.ndarray, q_medidos: np.ndarray, Pe: float, param1_guess: float, param2_guess: float, param2_conhecido: float = None) -> CalibrationResult:
        if len(pwf_medidos) < 3:
            raise ValueError("Calibração multiparâmetro exige pelo menos 3 pontos de teste.")

        # Identifica se estamos rodando Fetkovich ou Darcy-Vogel
        is_fetkovich = "Fetkovich" in self.strategy.get_model_name()

        # Define os limites matemáticos (Bounds) baseados na física do modelo
        if is_fetkovich:
            bounds_2d = ([1e-8, 0.5], [np.inf, 1.0]) # C > 0, n entre 0.5 e 1.0
            bounds_1d = ([1e-8], [np.inf])
        else:
            bounds_2d = ([1e-5, 1e-5], [np.inf, Pe]) # J > 0, Psat <= Pe
            bounds_1d = ([1e-5], [np.inf])

        if param2_conhecido is not None:
            # Otimização 1D: Apenas P1 (J ou C) é variável
            def residuals_1d(p1_array, pwf, q, pe):
                return self.strategy.residuals([p1_array[0], param2_conhecido], pwf, q, pe)

            resultado = least_squares(
                residuals_1d,
                x0=[param1_guess],
                args=(pwf_medidos, q_medidos, Pe),
                bounds=bounds_1d,
                method='trf'
            )
            p1_opt = resultado.x[0]
            p2_opt = param2_conhecido
        else:
            # Otimização 2D: Busca simultânea (J e Psat) ou (C e n)
            resultado = least_squares(
                self.strategy.residuals,
                x0=[param1_guess, param2_guess],
                args=(pwf_medidos, q_medidos, Pe),
                bounds=bounds_2d,
                method='trf'
            )
            p1_opt, p2_opt = resultado.x
            
        residuos_finais = resultado.fun
        q_preditos = q_medidos + residuos_finais
        
        rmse = np.sqrt(np.mean(residuos_finais**2))
        
        # Prevenção contra divisão por zero no MAPE
        mask_nonzero = q_medidos > 1e-5
        mape = mean_absolute_percentage_error(q_medidos[mask_nonzero], q_preditos[mask_nonzero]) if np.any(mask_nonzero) else 0.0

        result = CalibrationResult(
            well_name=well_name,
            model=self.strategy.get_model_name(),
            J_calibrado=p1_opt,    # Atua como J (Darcy) ou C (Fetkovich)
            Psat_calibrado=p2_opt, # Atua como Psat (Darcy) ou n (Fetkovich)
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
        logger.info(f"[{well_name}] Calibração salva. Modelo: {self.strategy.get_model_name()} | P1: {p1_opt:.3f} | RMSE: {rmse:.2f}")
        return result

# ==============================================================================
# NOVA FUNÇÃO DE DIAGNÓSTICO (Superfície RMSE Híbrida)
# ==============================================================================

def generate_rmse_surface(pwf_medidos, q_medidos, Pe, p1_opt, p2_opt, is_fetkovich=False, n_grid=50, delta_psi=5.0):
    # Cria a malha dependendo do modelo selecionado
    if is_fetkovich:
        p1_min, p1_max = p1_opt * 0.1, p1_opt * 2.0  # C
        p2_min, p2_max = 0.5, 1.0                    # n (Restrição física estrita)
    else:
        p1_min, p1_max = max(0.1, p1_opt * 0.5), min(10.0, p1_opt * 1.5)
        p2_min, p2_max = max(100.0, p2_opt * 0.7), min(Pe * 0.999, p2_opt * 1.2)

    P1_vals = np.linspace(p1_min, p1_max, n_grid)
    P2_vals = np.linspace(p2_min, p2_max, n_grid)
    P1_grid, P2_grid = np.meshgrid(P1_vals, P2_vals)
    
    MSE_grid = np.zeros_like(P1_grid)
    invalid_mask = np.zeros_like(P1_grid, dtype=bool)

    for pwf_real, q_real in zip(pwf_medidos, q_medidos):
        pwf_calc = np.full_like(P1_grid, np.nan)
        
        if is_fetkovich:
            # Equação inversa de Fetkovich para achar Pwf
            term = (q_real / P1_grid) ** (1.0 / P2_grid)
            inside_sqrt = (Pe**2) - term
            
            # Se inside_sqrt < 0, a combinação de C e n não consegue produzir essa vazão (Fisicamente impossível)
            invalid_mask |= (inside_sqrt < 0)
            
            mask_valid = inside_sqrt >= 0
            pwf_calc[mask_valid] = np.sqrt(inside_sqrt[mask_valid])
        else:
            # Lógica original Darcy-Vogel
            q_sat_grid = P1_grid * (Pe - P2_grid)
            
            # Darcy
            mask_darcy = q_real <= q_sat_grid
            pwf_calc[mask_darcy] = Pe - (q_real / P1_grid[mask_darcy])
            
            # Vogel
            mask_vogel = ~mask_darcy
            q_max_vogel = (P1_grid[mask_vogel] * P2_grid[mask_vogel]) / 1.8
            
            a, b = 0.8, 0.2
            c = ((q_real - q_sat_grid[mask_vogel]) / q_max_vogel) - 1.0
            delta = b**2 - 4*a*c
            
            invalid_mask[mask_vogel] |= (delta < 0)
            mask_valid_delta = delta >= 0
            delta_valid = delta[mask_valid_delta]
            
            x1 = (-b + np.sqrt(delta_valid)) / (2 * a)
            x2 = (-b - np.sqrt(delta_valid)) / (2 * a)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_valid_partial = np.where((x1 >= 0) & (x1 <= 1), x1,
                                   np.where((x2 >= 0) & (x2 <= 1), x2, np.nan))
            
            x_valid_full = np.full_like(delta, np.nan)
            x_valid_full[mask_valid_delta] = x_valid_partial
            
            invalid_mask[mask_vogel] |= np.isnan(x_valid_full)
            
            pwf_vogel_valid = np.full_like(delta, np.nan)
            pwf_vogel_valid[mask_valid_delta] = P2_grid[mask_vogel][mask_valid_delta] * x_valid_partial
            pwf_calc[mask_vogel] = pwf_vogel_valid

        MSE_grid[~invalid_mask] += (pwf_real - pwf_calc[~invalid_mask])**2

    RMSE_grid = np.sqrt(MSE_grid / len(q_medidos))
    RMSE_grid[invalid_mask] = np.nan

    rmse_min = np.nanmin(RMSE_grid)
    limiar_incerteza = rmse_min + delta_psi

    mask_valid_domain = ~np.isnan(RMSE_grid)
    mask_incerteza = (RMSE_grid <= limiar_incerteza) & mask_valid_domain
    area_incerteza_pct = (np.sum(mask_incerteza) / np.sum(mask_valid_domain)) * 100

    p1_incerteza = P1_grid[mask_incerteza]
    p2_incerteza = P2_grid[mask_incerteza]
    
    condicionamento_ci = np.nan
    if len(p1_incerteza) > 2:
        p1_norm = (p1_incerteza - np.mean(p1_incerteza)) / (np.std(p1_incerteza) + 1e-8)
        p2_norm = (p2_incerteza - np.mean(p2_incerteza)) / (np.std(p2_incerteza) + 1e-8)
        
        cov_matrix = np.cov(p1_norm, p2_norm)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        
        if len(eigenvalues) == 2:
            lambda_max = np.max(eigenvalues)
            lambda_min = np.min(eigenvalues)
            if lambda_min > 1e-8:
                condicionamento_ci = lambda_max / lambda_min

    # Os nomes das chaves são mantidos como J e Psat para não quebrar a interface gráfica do app.py,
    # mas representam C e n quando o modelo Fetkovich é ativado.
    return {
        "J_grid": P1_grid,
        "Psat_grid": P2_grid,
        "RMSE_grid": RMSE_grid,
        "rmse_min": rmse_min,
        "limiar_incerteza": limiar_incerteza,
        "area_incerteza_pct": area_incerteza_pct,
        "condicionamento_ci": condicionamento_ci
    }