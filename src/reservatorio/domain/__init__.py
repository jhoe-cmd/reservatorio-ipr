"""
Camada de Domínio (Domain Layer)
Contém as entidades puras, contratos de dados, regras físicas, matemática do reservatório
e as interfaces (Strategies) independentes de qualquer framework externo.
"""

# Entidades e Contratos de Validação
from .models import PocoFisico, CalibrationResult

# Regras Físicas Analíticas
from .skin import calcular_skin

# Estratégias de Curvas IPR (ATUALIZADO PARA A NOVA ARQUITETURA)
from .ipr_models import IPRStrategy, ModelosIPR

# Estratégias de Calibração (Cálculo de Resíduos)
from .calibration import CalibrationStrategy, DarcyVogelCalibration

# Estratégias Estocásticas (Distribuições de Probabilidade)
from .distributions import (
    DistributionStrategy,
    NormalDistribution,
    LogNormalDistribution,
    TriangularDistribution,
)

# Definição estrita da API pública do Domínio
__all__ = [
    # Models
    "PocoFisico",
    "CalibrationResult",
    
    # Skin
    "calcular_skin",
    
    # IPR
    "IPRStrategy",
    "ModelosIPR",  # <-- Nova classe central de equações
    
    # Calibration
    "CalibrationStrategy",
    "DarcyVogelCalibration",
    
    # Distributions
    "DistributionStrategy",
    "NormalDistribution",
    "LogNormalDistribution",
    "TriangularDistribution",
]