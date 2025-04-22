import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_excel("data/dados.xlsx")
        if 'Date' in df.columns:
            df['Data'] = pd.to_datetime(df['Date'])
        elif 'Data' in df.columns:
            df['Data'] = pd.to_datetime(df['Data'])
        return df.drop(columns='Date')
    except FileNotFoundError:
        # Se o arquivo não for encontrado, criar dados de exemplo
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="M")
        df = pd.DataFrame({
            "Data": dates,
            "Receita_Transporte": np.random.normal(1000000, 200000, len(dates)),
            "Despesa_Transporte": np.random.normal(900000, 150000, len(dates))
        })
        df["Saldo_Transporte"] = df["Receita_Transporte"] - df["Despesa_Transporte"]
        return df