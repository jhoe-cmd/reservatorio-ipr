"""
Reservatório IPR
Biblioteca científica para modelagem e calibração de produtividade de poços.
"""

__version__ = "3.1.0"

# Exposição da API Pública do Pacote
from .config import ReservoirConstants
from .domain.models import PocoFisico, CalibrationResult
from .domain.skin import calcular_skin
from .domain.ipr_models import ModelosIPR
from .domain.calibration import DarcyVogelCalibration
from .domain.distributions import NormalDistribution, LogNormalDistribution, TriangularDistribution
from .infrastructure.repositories import JsonCalibrationRepository
from .application.services import AnaliseProdutividadeService
from .application.optimization import HistoryMatchingService
from .application.montecarlo import MonteCarloIPR

# Define o que é exportado quando alguém faz "from reservatorio import *"
__all__ = [
    "__version__",
    "ReservoirConstants",
    "PocoFisico",
    "CalibrationResult",
    "calcular_skin",
    "DarcyVogelHibridoIPR",
    "fator_vogel_math",
    "DarcyVogelCalibration",
    "NormalDistribution",
    "LogNormalDistribution",
    "TriangularDistribution",
    "JsonCalibrationRepository",
    "AnaliseProdutividadeService",
    "HistoryMatchingService",
    "MonteCarloIPR",
]