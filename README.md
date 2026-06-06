# 🛢️ Reservatório IPR (Inflow Performance Relationship)

**Versão:** 3.1.0  
**Autor:** Márcio Gomes da Silva (UFAL)

Biblioteca científica avançada em Python para modelagem, calibração (History Matching) e análise de incertezas (Monte Carlo) da produtividade de poços de petróleo e gás.

A arquitetura do projeto foi construída utilizando os princípios de **Domain-Driven Design (DDD)**, garantindo a separação estrita entre a matemática pura, as validações termodinâmicas e a orquestração de serviços.

---

## 🚀 Principais Funcionalidades

* **Segurança Dimensional:** Validação estrita de unidades físicas utilizando `pint` e `pydantic`.
* **Calibração Multiparâmetro (History Matching):** Ajuste automático do Índice de Produtividade ($J$) e da Pressão de Saturação ($P_{sat}$) minimizando o RMSE contra dados reais de testes *Multirate* (via `scipy.optimize`).
* **Análise de Incertezas (Monte Carlo):** Simulação estocástica 100% vetorizada para extração de percentis de risco (P10, P50, P90) da AOF.
* **Persistência Científica:** Geração de *logs* auditáveis com métricas estatísticas detalhadas (RMSE, MAE, MAPE, R², Bias).
* **Robustez Matemática:** Cobertura de testes baseada em propriedades (via `hypothesis`) e validação analítica da continuidade ($C^0$ e $C^1$) no ponto de bolha.

---

## ⚙️ Instalação e Execução

Recomenda-se a utilização de um ambiente virtual (venv ou conda). Para instalar o pacote no modo de desenvolvimento interativo (com todas as dependências de engenharia e testes):

```bash
# Na raiz do projeto (onde está o pyproject.toml)
pip install -e .[dev]