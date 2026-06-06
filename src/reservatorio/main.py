import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from reservatorio.domain.ipr_models import DarcyVogelHibridoIPR
from reservatorio.domain.calibration import DarcyVogelCalibration
from reservatorio.domain.distributions import NormalDistribution, LogNormalDistribution
from reservatorio.infrastructure.repositories import JsonCalibrationRepository
from reservatorio.application.optimization import HistoryMatchingService
from reservatorio.application.montecarlo import MonteCarloIPR

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def coletar_dados_interativos():
    """Função para coletar os dados do usuário direto pelo terminal."""
    print("\n" + "="*55)
    print("🛢️  SIMULADOR IPR - ENTRADA DE DADOS DE RESERVATÓRIO")
    print("="*55)
    
    try:
        nome = input("Nome do Poço (ex: Poço Bravo): ") or "Poço_Desconhecido"
        pe = float(input("Pressão Estática do Reservatório - Pe (psi): "))
        
        print("\n--- Dados do Teste de Produção (Separador) ---")
        print("Digite os valores separados por VÍRGULA (ex: 3500, 3000, 2500)")
        pwf_str = input("Pressões de Fundo - Pwf (psi): ")
        q_str = input("Vazões Correspondentes - Q (STB/d): ")
        
        # Converte as strings digitadas em Arrays do NumPy
        pwf_campo = np.array([float(x.strip()) for x in pwf_str.split(',')])
        q_campo = np.array([float(x.strip()) for x in q_str.split(',')])
        
        if len(pwf_campo) != len(q_campo):
            logger.error("A quantidade de pressões e vazões deve ser rigorosamente igual!")
            sys.exit(1)
            
        print("\n--- Chutes Iniciais para Otimização (History Matching) ---")
        j_guess = float(input("Chute inicial para o Índice de Produtividade J (ex: 1.5): "))
        psat_guess = float(input("Chute inicial para a Pressão de Bolha Psat (ex: 2000): "))
        
        return nome, pe, pwf_campo, q_campo, j_guess, psat_guess
        
    except ValueError:
        logger.error("Erro de digitação. Por favor, reinicie e insira apenas números e vírgulas válidas.")
        sys.exit(1)

def main():
    # Menu principal
    print("\nEscolha o modo de execução:")
    print("1. Rodar os dados da Questão 1 (Automático)")
    print("2. Inserir novos dados de campo manualmente")
    modo = input("Digite 1 ou 2: ").strip()
    
    if modo == "2":
        well_name, Pe_campo, pwf_campo, q_campo, J_guess, Psat_guess = coletar_dados_interativos()
    else:
        logger.info("Carregando dados pré-configurados...")
        well_name = "Poço Alpha"
        Pe_campo = 4000.0
        pwf_campo = np.array([3500.0, 3000.0, 2500.0, 1500.0])
        q_campo = np.array([800.0, 1550.0, 2200.0, 3100.0])
        J_guess = 1.5
        Psat_guess = 2000.0

    print("\n" + "-"*55)
    logger.info("Iniciando Motor de Simulação...")
    
    repo = JsonCalibrationRepository()
    calibrador = HistoryMatchingService(strategy=DarcyVogelCalibration(), repository=repo)

    # 1. Calibração
    res_calibracao = calibrador.calibrar(
        well_name=well_name,
        pwf_medidos=pwf_campo,
        q_medidos=q_campo,
        Pe=Pe_campo,
        J_guess=J_guess,
        Psat_guess=Psat_guess
    )
    
    # 2. Monte Carlo
    simulador_mc = MonteCarloIPR()
    risco = simulador_mc.run(
        pe_dist=NormalDistribution(mean=Pe_campo, std=Pe_campo*0.05),
        psat_dist=NormalDistribution(mean=res_calibracao.Psat_calibrado, std=150.0),
        j_dist=LogNormalDistribution(mean=np.log(res_calibracao.J_calibrado), sigma=0.15),
        n_simulations=50000
    )
    
    logger.info(f"Monte Carlo AOF: P90={risco['P90_Conservador']:.0f} | P50={risco['P50_Esperado']:.0f} | P10={risco['P10_Otimista']:.0f}")

    # 3. Plotagem Dinâmica
    modelo = DarcyVogelHibridoIPR()
    class MockPoco:
        Pe = Pe_campo
        Psat = res_calibracao.Psat_calibrado
        q_test = q_campo[1] if len(q_campo) > 1 else q_campo[0]
        Pwf_test = pwf_campo[1] if len(pwf_campo) > 1 else pwf_campo[0]
        
    q_arr, pwf_arr, _, aof = modelo.calcular_curva(MockPoco(), J_in=res_calibracao.J_calibrado)

    plt.figure(figsize=(9, 5))
    plt.plot(q_arr, pwf_arr, 'b-', linewidth=2, label=f'IPR Calibrada (AOF: {aof:.0f})')
    plt.scatter(q_campo, pwf_campo, color='red', zorder=5, label='Dados de Teste')
    plt.title(f'Calibração IPR - {well_name}', fontweight='bold')
    plt.xlabel('Vazão (STB/dia)', fontweight='bold')
    plt.ylabel('Pressão de Fundo - Pwf (psi)', fontweight='bold')
    plt.ylim(0, Pe_campo + 500)
    plt.xlim(0, aof * 1.1)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    # Salva o arquivo dinamicamente com o nome do poço
    nome_arquivo = f'grafico_{well_name.replace(" ", "_")}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    logger.info(f"Sucesso! Gráfico salvo como '{nome_arquivo}'.")

if __name__ == "__main__":
    main()