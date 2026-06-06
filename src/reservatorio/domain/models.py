from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

class PocoFisico(BaseModel):
    """Modelo de domínio do Poço com segurança dimensional."""
    nome: str = Field(..., min_length=1)
    Pe_str: str = Field(..., description="Pressão do Reservatório (ex: '4000 psi')")
    Psat_str: str = Field(..., description="Pressão de Saturação (ex: '2000 psi')")
    q_test: float = Field(..., gt=0)
    Pwf_test: float = Field(..., ge=0)

    Pe: float = 0.0
    Psat: float = 0.0

    @model_validator(mode='after')
    def validar_unidades_e_termodinamica(self) -> 'PocoFisico':
        pe_qty = Q_(self.Pe_str).to(ureg.psi)
        psat_qty = Q_(self.Psat_str).to(ureg.psi)

        self.Pe = pe_qty.magnitude
        self.Psat = psat_qty.magnitude

        if self.Pe <= 0: raise ValueError("Pe deve ser > 0.")
        if self.Pwf_test >= self.Pe: raise ValueError("Pwf_test >= Pe.")
        if self.Psat > self.Pe: raise ValueError("Psat > Pe.")
        return self

class CalibrationResult(BaseModel):
    """Modelo de persistência científica exaustivo para auditoria."""
    well_name: str
    date: datetime = Field(default_factory=datetime.utcnow)
    model: str
    
    # Parâmetros
    J_calibrado: float
    Psat_calibrado: float | None = None
    
    # Métricas Estatísticas
    rmse: float
    mae: float
    mape: float
    r2: float
    bias: float
    
    # Metadados do Otimizador (SciPy)
    success: bool
    nfev: int
    cost: float
    message: str
    
    version: str = "3.1.0"