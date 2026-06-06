import numpy as np
from typing import Dict
from reservatorio.config import ReservoirConstants
from reservatorio.domain.distributions import DistributionStrategy

class MonteCarloIPR:
    """Simulação Estocástica 100% vetorizada com reprodutibilidade."""
    
    def __init__(self, seed: int = ReservoirConstants.DEFAULT_SEED):
        self.rng = np.random.default_rng(seed=seed)

    def run(self, 
            pe_dist: DistributionStrategy, 
            psat_dist: DistributionStrategy, 
            j_dist: DistributionStrategy, 
            n_simulations: int = 10000) -> Dict[str, float]:
        
        pe_array = pe_dist.sample(n_simulations, self.rng)
        psat_array = psat_dist.sample(n_simulations, self.rng)
        j_array = j_dist.sample(n_simulations, self.rng)
        
        pe_array = np.clip(pe_array, 1e-5, None)
        psat_array = np.clip(psat_array, 1e-5, pe_array)
        j_array = np.clip(j_array, 1e-5, None)
        
        # Cálculo 100% vetorizado (eliminação do laço for)
        qb = j_array * (pe_array - psat_array)
        aof_results = qb + (j_array * psat_array) / ReservoirConstants.VOGEL_CONSTANT

        p90, p50, p10 = np.percentile(aof_results, [10, 50, 90])
        
        return {
            "P90_Conservador": float(p90),
            "P50_Esperado": float(p50),
            "P10_Otimista": float(p10),
            "Media": float(np.mean(aof_results)),
            "Desvio_Padrao": float(np.std(aof_results))
        }