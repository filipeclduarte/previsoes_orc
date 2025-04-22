import plotly.express as px

def plot_temporal(df, categoria):
    fig = px.line(df, x="Data", y=categoria, title=f"{categoria} ao Longo do Tempo",
                  template="plotly_dark", line_shape="spline")
    fig.update_layout(showlegend=True, xaxis_title="Data", yaxis_title="Valor (R$)")
    return fig

def plot_monthly(df_monthly):
    fig = px.bar(df_monthly, x="Data", y=["Receita_Transporte", "Despesa_Transporte", "Saldo_Transporte"],
                 title="Acumulado Mensal", template="plotly_dark", barmode="group")
    return fig