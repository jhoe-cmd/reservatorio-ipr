from abc import ABC, abstractmethod
import numpy as np

class DistributionStrategy(ABC):
    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        pass

class NormalDistribution(DistributionStrategy):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)

class LogNormalDistribution(DistributionStrategy):
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mean, self.sigma, n)

class TriangularDistribution(DistributionStrategy):
    def __init__(self, left: float, mode: float, right: float):
        self.left = left
        self.mode = mode
        self.right = right
        
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.left, self.mode, self.right, n)