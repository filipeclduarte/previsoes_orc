import streamlit as st
import pandas as pd

# Enable hot reload
st._config.set_option("server.runOnSave", True)

from data.load_data import load_data
from modules.visualizations import plot_temporal
from modules.predictions import generate_forecast
from modules.utils import format_currency, download_csv

# Configuração do Streamlit
st.set_page_config(page_title="Dashboard Orçamentário", layout="wide")

# CSS customizado para responsividade mobile
def local_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

responsive_css = '''
@media (max-width: 600px) {
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    .css-18e3th9, .css-1d391kg { /* Main container */
        padding: 0 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-size: 1.1em !important;
    }
    button, .stButton>button {
        font-size: 1em !important;
        padding: 0.4em 0.6em !important;
    }
    .stSidebar {
        width: 80vw !important;
        min-width: 120px !important;
    }
    .stDataFrame, .stTable {
        font-size: 0.8em !important;
    }
}
'''
local_css(responsive_css)

st.title("Dashboard Interativo de Dados Orçamentários")

# Carregar dados
df = load_data()

# Sidebar para filtros
st.sidebar.header("Filtros")
ano = 2025
# ano = st.sidebar.slider("Selecione o Ano", min_value=int(df["Data"].dt.year.min()), 
                        # max_value=int(df["Data"].dt.year.max()), value=int(df["Data"].dt.year.max()))
# )
categoria = st.sidebar.selectbox("Selecione a Categoria", ["Hospital do Servidor Público Municipal", "Fundo Municipal de Saúde", "Secretaria Municipal de Educação"])

# Permitir seleção de múltiplos modelos
modelos_disponiveis = [
    "Prophet",
    "AutoARIMA",
    "AutoETS",
    # "MLP",  # Desativado temporariamente
    # "LSTM"  # Desativado temporariamente
]
modelos = st.sidebar.multiselect("Selecione os Modelos de Previsão", modelos_disponiveis, default=["Prophet"])

# Filtrar dados
df_filtered = df[df["Data"].dt.year == ano]

# Módulo 1: Visualizações Temporais
st.header("Visualização Temporal")
fig_temporal = plot_temporal(df_filtered, categoria)
st.plotly_chart(fig_temporal, use_container_width=True)

# Módulo 2: Tabela Interativa e Download
st.header("Tabela de Dados")
if st.checkbox("Mostrar Tabela"):
    st.dataframe(df_filtered.style.format({
        "Hospital do Servidor Público Municipal": format_currency,
        "Fundo Municipal de Saúde": format_currency,
        "Secretaria Municipal de Educação": format_currency
    }))

download_csv(df_filtered, f"dados_{ano}.csv", "Download dos Dados (CSV)")

# Módulo 3: Previsões
st.header("Previsões Orçamentárias")

# Previsão sob demanda
if not modelos:
    st.warning("Por favor, selecione ao menos um modelo no sidebar para exibir previsões.")
else:
    gerar = st.button('Gerar Previsão')
    if gerar:
        try:
            with st.spinner('Gerando previsões, aguarde...'):
                fig_forecast, forecasts_dict, metrics_df, tabela_previsoes, melhor_modelo, metrics_df_ordenado = generate_forecast(
                    df_filtered, categoria, modelos, data_final='2025-04-24'
                )
                st.session_state.forecasts_dict = forecasts_dict
                st.session_state.metrics_df = metrics_df
                st.session_state.fig_forecast = fig_forecast
                st.session_state.tabela_previsoes = tabela_previsoes
                st.session_state.melhor_modelo = melhor_modelo
                st.session_state.metrics_df_ordenado = metrics_df_ordenado
        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")
            st.session_state.forecasts_dict = None
            st.session_state.metrics_df = None
            st.session_state.fig_forecast = None
            st.session_state.tabela_previsoes = None

if st.session_state.get("fig_forecast") is not None and st.session_state.get("forecasts_dict") is not None:
    any_nonempty = any([not df.empty for df in st.session_state["forecasts_dict"].values() if df is not None])
    if any_nonempty:
        st.plotly_chart(st.session_state.get("fig_forecast"), use_container_width=True)
    else:
        st.warning("Nenhuma previsão foi gerada para os modelos selecionados.")

# Aviso visual para modelos sem previsão
if st.session_state.get("forecasts_dict") is not None:
    for modelo in modelos:
        forecast_df = st.session_state["forecasts_dict"].get(modelo)
        if forecast_df is not None and forecast_df.empty:
            st.warning(f"O modelo {modelo} não conseguiu gerar previsões para este conjunto de dados.")

# Criar um arquivo CSV combinado com todas as previsões
if st.session_state.get("forecasts_dict") is not None:
    all_forecasts = pd.DataFrame()
    for model_name, forecast_df in st.session_state.forecasts_dict.items():
        if not forecast_df.empty:
            df_copy = forecast_df.copy()
            df_copy['modelo'] = model_name
            all_forecasts = pd.concat([all_forecasts, df_copy], ignore_index=True)
    if not all_forecasts.empty:
        download_csv(
            all_forecasts,
            f"previsoes_{categoria.lower()}_{ano}.csv",
            "Download das Previsões (CSV)"
        )

# Checkbox para mostrar a tabela de previsões
show_forecast_table = st.checkbox("Mostrar Tabela de Previsões Consolidada")
if show_forecast_table and st.session_state.get("tabela_previsoes") is not None:
    tabela = st.session_state.get("tabela_previsoes").copy()
    if not tabela.empty:
        # Formatar datas
        if 'ds' in tabela.columns:
            tabela['Data'] = tabela['ds'].dt.strftime('%Y-%m-%d')
            tabela = tabela.drop(columns=['ds'])
        # Formatar valores monetários
        cols_currency = [col for col in tabela.columns if 'Previsão' in col or 'Valor Real' in col]
        for col in cols_currency:
            tabela[col] = tabela[col].apply(lambda x: format_currency(x) if pd.notnull(x) else '-')
        st.dataframe(tabela, use_container_width=True)
    else:
        st.warning("Tabela de previsões está vazia.")

# Módulo para mostrar métricas de avaliação
st.header("Métricas de Avaliação dos Modelos")
if st.session_state.get("metrics_df") is not None:
    df_metrics = st.session_state.get("metrics_df")
    if not df_metrics.empty:
        # Formatar a tabela de métricas
        st.dataframe(df_metrics.style.format({
            "RMSE": lambda x: f"{x:.2f}",
            "MAE": lambda x: f"{x:.2f}",
            "MAPE": lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
        }))
        # Módulo visual para melhor modelo (usando score_total)
        if st.session_state.get("metrics_df_ordenado") is not None:
            df_ord = st.session_state.metrics_df_ordenado
            best_row = df_ord.iloc[0]
            st.markdown(f"""
                <div style='padding: 1em; background: linear-gradient(90deg, #e0ffe0 0%, #b2f7b2 100%); border-radius: 10px; margin-top: 20px; text-align: center; font-size: 1.2em;'>
                <span style='font-size:2em;'>🏆</span><br>
                <b>Melhor modelo para este conjunto de dados:</b><br>
                <span style='font-size:1.3em; color:#2e7d32;'><b>{best_row['Modelo']}</b></span><br>
                <span style='font-size:1em;'>Score Total = <b>{best_row['score_total']:.2f}</b></span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Tabela de métricas de erro está vazia.")
