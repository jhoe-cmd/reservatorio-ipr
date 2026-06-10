"""
Camada de Aplicação (Application Layer)
Orquestra os casos de uso da engenharia de reservatórios, coordenando as regras de domínio e a infraestrutura.
"""

#from .services import AnaliseProdutividadeService
from .optimization import HistoryMatchingService
from .montecarlo import MonteCarloIPR

# Define o que fica disponível quando se faz: from reservatorio.application import *
__all__ = [
    "AnaliseProdutividadeService",
    "HistoryMatchingService",
    "MonteCarloIPR",
]