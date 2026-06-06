"""
Camada de Infraestrutura (Infrastructure Layer)
Responsável por isolar os detalhes técnicos de persistência de dados (I/O), 
comunicação com o sistema de ficheiros, bases de dados e APIs externas.
"""

from .repositories import CalibrationRepository, JsonCalibrationRepository

# Define a API pública da camada de infraestrutura
__all__ = [
    "CalibrationRepository",
    "JsonCalibrationRepository",
]