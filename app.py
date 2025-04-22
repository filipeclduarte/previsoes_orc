import streamlit as st
import pandas as pd

# Enable hot reload
st._config.set_option("server.runOnSave", True)

from data.load_data import load_data
from modules.visualizations import plot_temporal, plot_monthly
from modules.predictions import generate_forecast
from modules.report import generate_pdf_report
from modules.utils import format_currency, download_csv

# Configurau00e7u00e3o do Streamlit
st.set_page_config(page_title="Dashboard Oru00e7amentu00e1rios", layout="wide")
st.title("Dashboard Interativo de Dados Oru00e7amentu00e1rios")

# Carregar dados
df = load_data()

# Sidebar para filtros
st.sidebar.header("Filtros")
ano = 2025
# ano = st.sidebar.slider("Selecione o Ano", min_value=int(df["Data"].dt.year.min()), 
                        # max_value=int(df["Data"].dt.year.max()), value=int(df["Data"].dt.year.max()))
# )
categoria = st.sidebar.selectbox("Selecione a Categoria", ["Hospital do Servidor Pu00fablico Municipal", "Fundo Municipal de Sau00fade", "Secretaria Municipal de Educau00e7u00e3o"])

# Permitir seleu00e7u00e3o de mu00faltiplos modelos
modelos_disponiveis = ["Prophet", "AutoARIMA", "AutoETS", "MLP", "LSTM"]
modelos = st.sidebar.multiselect("Selecione os Modelos de Previsu00e3o", modelos_disponiveis, default=["Prophet"])

# Filtrar dados
df_filtered = df[df["Data"].dt.year == ano]

# Mu00f3dulo 1: Visualizau00e7u00f5es Temporais
st.header("Visualizau00e7u00e3o Temporal")
fig_temporal = plot_temporal(df_filtered, categoria)
st.plotly_chart(fig_temporal, use_container_width=True)

# Mu00f3dulo 2: Tabela Interativa e Download
st.header("Tabela de Dados")
if st.checkbox("Mostrar Tabela"):
    st.dataframe(df_filtered.style.format({
        "Hospital do Servidor Pu00fablico Municipal": format_currency,
        "Fundo Municipal de Sau00fade": format_currency,
        "Secretaria Municipal de Educau00e7u00e3o": format_currency
    }))

download_csv(df_filtered, f"dados_{ano}.csv", "Download dos Dados (CSV)")

# Mu00f3dulo 3: Previsu00f5es
st.header("Previsu00f5es Oru00e7amentu00e1rias")

# Variu00e1veis para armazenar os resultados da previsu00e3o
if 'forecasts_dict' not in st.session_state:
    st.session_state.forecasts_dict = None
    
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None

# Verificar se pelo menos um modelo foi selecionado
if not modelos:
    st.warning("Por favor, selecione pelo menos um modelo de previsão.")
else:
    if st.button("Gerar Previsões"):
        try:
            fig_forecast, forecasts_dict, metrics_df = generate_forecast(df, categoria, modelos, future_days=7)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Armazenar os resultados na session_state
            st.session_state.forecasts_dict = forecasts_dict
            st.session_state.metrics_df = metrics_df
            
            # Criar um arquivo CSV combinado com todas as previsões
            all_forecasts = pd.DataFrame()
            for modelo, forecast_df in forecasts_dict.items():
                if len(forecast_df) > 0:
                    forecast_df = forecast_df.copy()
                    forecast_df['modelo'] = modelo
                    all_forecasts = pd.concat([all_forecasts, forecast_df])
            
            if len(all_forecasts) > 0:
                download_csv(all_forecasts, f"previsoes_{categoria.lower()}_{ano}.csv", 
                            "Download das Previsões (CSV)")
        except Exception as e:
            st.error(f"Erro ao gerar previsu00e3o: {e}")
            st.info("Tente outro modelo ou categoria de dados.")

# Checkbox para mostrar a tabela de previsões
show_forecast_table = st.checkbox("Mostrar Tabela de Previsões")
if show_forecast_table and st.session_state.forecasts_dict is not None:
    # Verificar se há modelos com previsões
    modelos_com_previsoes = [modelo for modelo, df in st.session_state.forecasts_dict.items() if len(df) > 0]
    
    if modelos_com_previsoes:  # Verificar se a lista não está vazia
        # Criar tabs para cada modelo
        modelo_tabs = st.tabs(modelos_com_previsoes)
        
        for i, (modelo, tab) in enumerate(zip(modelos_com_previsoes, modelo_tabs)):
            with tab:
                forecast_df = st.session_state.forecasts_dict[modelo]
                if len(forecast_df) > 0:
                    # Formatar a tabela de previsões
                    st.dataframe(forecast_df.style.format({
                        "ds": lambda x: x.strftime("%Y-%m-%d"),
                        "yhat": format_currency,
                        "yhat_lower": format_currency,
                        "yhat_upper": format_currency
                    }))
                else:
                    st.info(f"Previsões disponíveis para o modelo {modelo}.")
    else:
        st.info("Não há previsões disponíveis. Tente gerar previsões primeiro.")

# Módulo para mostrar métricas de avaliação
st.header("Métricas de Avaliação dos Modelos")
if st.session_state.metrics_df is not None:
    # Formatar a tabela de métricas
    st.dataframe(st.session_state.metrics_df.style.format({
        "RMSE": lambda x: f"{x:.2f}",
        "MAE": lambda x: f"{x:.2f}",
        "MAPE": lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
    }))

