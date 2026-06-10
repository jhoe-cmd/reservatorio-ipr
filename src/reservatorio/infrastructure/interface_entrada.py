import streamlit as st
import pandas as pd

class InterfaceEntradaDados:
    """
    Camada de Infraestrutura: Gerencia a interface com o usuário para a 
    inserção de dados de campo, suportando upload de arquivos e edição interativa.
    """

    @staticmethod
    def renderizar_entrada_dados():
        st.subheader("📊 Dados de Teste do Poço")
        st.write("Insira os dados de Pressão de Fundo (Pwf) e Vazão medida.")

        # Opções de método de entrada
        metodo = st.radio(
            "Selecione o método de entrada:",
            ["Edição Manual / Colar do Excel", "Upload de Planilha (CSV/Excel)"],
            horizontal=True
        )

        df_final = None

        if metodo == "Upload de Planilha (CSV/Excel)":
            arquivo = st.file_uploader("Arraste seu arquivo CSV ou Excel aqui", type=['csv', 'xlsx'])
            
            if arquivo is not None:
                try:
                    # Verifica a extensão para usar o motor de leitura correto do Pandas
                    if arquivo.name.endswith('.csv'):
                        df_final = pd.read_csv(arquivo)
                    else:
                        df_final = pd.read_excel(arquivo)
                        
                    st.success(f"Arquivo '{arquivo.name}' carregado com sucesso!")
                    
                    # Mostra um preview dos dados carregados
                    with st.expander("Visualizar dados carregados"):
                        st.dataframe(df_final, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo. Verifique o formato. Detalhe: {e}")

        else:
            # Método de edição interativa
            st.info("💡 Dica: Você pode copiar os dados do seu Excel e colar com Ctrl+V diretamente na primeira célula da tabela abaixo.")
            
            # Cria um DataFrame vazio com a estrutura padrão esperada
            df_base = pd.DataFrame({
                "Pwf (psi)": [0.0, 0.0, 0.0, 0.0],
                "Vazão (bbl/d)": [0.0, 0.0, 0.0, 0.0]
            })
            
            # st.data_editor permite adicionar/remover linhas dinamicamente e colar dados
            df_final = st.data_editor(
                df_base, 
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True
            )

        return df_final

    @staticmethod
    def validar_dados(df):
        """
        Valida se o DataFrame retornado possui as colunas necessárias e 
        não contém valores nulos antes de enviar para o otimizador.
        """
        if df is None or df.empty:
            return False, None, None
            
        # Pega os arrays numpy a partir das colunas (assumindo que a primeira é Pressão e a segunda Vazão)
        try:
            pwf_array = df.iloc[:, 0].astype(float).values
            vazao_array = df.iloc[:, 1].astype(float).values
            
            # Filtra linhas onde os valores são zero (caso o usuário deixe as linhas padrão vazias)
            mask = (pwf_array > 0) | (vazao_array > 0)
            pwf_array = pwf_array[mask]
            vazao_array = vazao_array[mask]
            
            if len(pwf_array) < 3:
                st.warning("⚠️ Insira pelo menos 3 pontos de teste para que a calibração matemática seja confiável.")
                return False, None, None
                
            return True, pwf_array, vazao_array
            
        except Exception:
            st.error("Erro na validação: Certifique-se de que as colunas contêm apenas números.")
            return False, None, None