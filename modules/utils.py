import streamlit as st
import io

def format_currency(value):
    return f"R${value:,.2f}"

def download_csv(df, filename, label):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/csv"
    )