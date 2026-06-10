import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

class VisualizadorIncerteza:
    """
    Camada de Infraestrutura: Responsável pela renderização visual da topografia
    de erro, demonstrando visualmente a identificabilidade (ou a falta dela).
    """

    @staticmethod
    def gerar_malha_rmse(modelo_func, pe, pwf_medido, q_medido, j_range, psat_range):
        """
        Realiza uma varredura (grid search) calculando o RMSE para cada 
        combinação de J e Psat dentro dos limites fornecidos.
        """
        # Cria as matrizes vazias para a malha
        J_grid, Psat_grid = np.meshgrid(j_range, psat_range)
        RMSE_grid = np.zeros_like(J_grid)

        # Preenche a malha com os valores de RMSE
        for i in range(J_grid.shape[0]):
            for k in range(J_grid.shape[1]):
                j_atual = J_grid[i, k]
                psat_atual = Psat_grid[i, k]
                
                # Calcula a vazão teórica para esta combinação
                q_calculado = modelo_func(pwf_medido, pe, psat_atual, j_atual)
                
                # Calcula e armazena o RMSE
                rmse = np.sqrt(mean_squared_error(q_medido, q_calculado))
                RMSE_grid[i, k] = rmse
                
        return J_grid, Psat_grid, RMSE_grid

    @staticmethod
    def plotar_contorno_2d(J_grid, Psat_grid, RMSE_grid, j_otimo, psat_otima):
        """Gera o mapa de calor/contorno 2D da superfície de erro."""
        fig = go.Figure(data=go.Contour(
            z=RMSE_grid,
            x=J_grid[0], # Eixo X: J
            y=Psat_grid[:, 0], # Eixo Y: Psat
            colorscale='Viridis',
            colorbar=dict(title='RMSE'),
            contours=dict(showlines=False)
        ))

        # Adiciona uma estrela marcando o ponto ótimo encontrado pelo algoritmo
        fig.add_trace(go.Scatter(
            x=[j_otimo], y=[psat_otima],
            mode='markers',
            marker=dict(symbol='star', size=15, color='white', line=dict(color='black', width=2)),
            name='Ótimo Encontrado'
        ))

        fig.update_layout(
            title='Superfície de Erro 2D: Diagnóstico de Identificabilidade',
            xaxis_title='Índice de Produtividade - J',
            yaxis_title='Pressão de Saturação - Psat',
            template='plotly_white'
        )
        return fig

    @staticmethod
    def plotar_superficie_3d(J_grid, Psat_grid, RMSE_grid):
        """Gera a visualização 3D da topografia de erro para o usuário girar e inspecionar."""
        fig = go.Figure(data=[go.Surface(
            z=RMSE_grid, 
            x=J_grid, 
            y=Psat_grid,
            colorscale='Viridis'
        )])

        fig.update_layout(
            title='Topografia de Erro 3D',
            scene=dict(
                xaxis_title='J',
                yaxis_title='Psat',
                zaxis_title='RMSE'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig