import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DiagnosticoIdentificabilidade:
    """
    Camada de Aplicação: Avalia a qualidade estatística e matemática do ajuste.
    Calcula as métricas clássicas e o Índice de Condicionamento (CI) para 
    diagnosticar a não-identificabilidade do modelo.
    """

    @staticmethod
    def calcular_metricas_classicas(y_real, y_calculado):
        """Calcula RMSE, MAE e R2."""
        rmse = np.sqrt(mean_squared_error(y_real, y_calculado))
        mae = mean_absolute_error(y_real, y_calculado)
        r2 = r2_score(y_real, y_calculado)
        
        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4)
        }

    @staticmethod
    def calcular_indice_condicionamento(jacobiana):
        """
        Calcula o Índice de Condicionamento (CI) a partir da Jacobiana do otimizador.
        Na otimização de mínimos quadrados, a aproximação da Hessiana é J^T * J.
        O CI mede a razão entre o maior e o menor autovalor dessa matriz.
        """
        try:
            # Aproximação da Hessiana (J transposta * J)
            hessiana_aprox = np.dot(jacobiana.T, jacobiana)
            
            # Autovalores da matriz
            autovalores = np.linalg.eigvals(hessiana_aprox)
            
            # Garantir que não haja divisão por zero e pegar os valores absolutos
            autovalores = np.abs(autovalores)
            lambda_max = np.max(autovalores)
            lambda_min = np.min(autovalores)
            
            if lambda_min == 0:
                return float('inf') # Sistema singular (degenerescência total)
                
            ci = lambda_max / lambda_min
            return round(ci, 2)
            
        except Exception as e:
            return None

    @staticmethod
    def classificar_status_solucao(ci):
        """Gera o diagnóstico automático baseado no valor do CI."""
        if ci is None:
            return "Erro no cálculo da Jacobiana", "🔴"
        elif ci < 10:
            return "Excelente (Solução Robusta e Única)", "🟢"
        elif ci < 100:
            return "Aceitável (Média Correlação)", "🟡"
        elif ci < 1000:
            return "Fraco (Forte Correlação Paramétrica)", "🟠"
        else:
            return "Não Identificável (Degenerescência Geométrica)", "🔴"

    @classmethod
    def gerar_relatorio_completo(cls, y_real, y_calculado, jacobiana):
        """Orquestra as funções acima para entregar o relatório final."""
        metricas = cls.calcular_metricas_classicas(y_real, y_calculado)
        ci = cls.calcular_indice_condicionamento(jacobiana)
        status_texto, status_cor = cls.classificar_status_solucao(ci)
        
        relatorio = {
            "Metricas": metricas,
            "CI": ci,
            "Status_Texto": status_texto,
            "Status_Cor": status_cor
        }
        return relatorio