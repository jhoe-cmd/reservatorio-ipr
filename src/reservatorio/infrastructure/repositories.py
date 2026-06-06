import json
import os
from abc import ABC, abstractmethod
from reservatorio.domain.models import CalibrationResult

class CalibrationRepository(ABC):
    @abstractmethod
    def save(self, result: CalibrationResult) -> None:
        pass

class JsonCalibrationRepository(CalibrationRepository):
    def __init__(self, filepath: str = "historico_calibracao.json"):
        self.filepath = filepath

    def save(self, result: CalibrationResult) -> None:
        data = []
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        data.append(result.model_dump(mode='json'))
        
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)