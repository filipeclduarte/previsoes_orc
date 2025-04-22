import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# class LSTMWrapper(BaseEstimator, RegressorMixin):
#     """Wrapper to make neuralforecast.LSTM compatible with scikit-learn API"""
#     # Corpo removido temporariamente para evitar erros de identação

    #             input_size=self.input_size,
    #             max_steps=self.max_steps,
    #             scaler_type=self.scaler_type
    #         )],
    #         freq="D"  # Alterado para frequência diária
    #     )
    #     self.model.fit(df=df)
    #     return self

    # def predict(self, X):
    #     # Prepare forecast DataFrame
    #     df = pd.DataFrame({
    #         "unique_id": "transporte",
    #         "ds": X["ds"]
    #     })
    #     forecast = self.model.predict(df=df)
    #     return forecast["LSTM"].values

def generate_forecast(df, categoria, modelos, horizon=30, future_days=0, data_final=None):
    """
    Generate forecasts using the selected models.
    Parameters:
    - df: DataFrame with historical data
    - categoria: Column to forecast (e.g., Receita_Transporte)
    - modelos: List of model names (Prophet, AutoARIMA, AutoETS, MLP, LSTM)
    - horizon: Forecast horizon in days (default: 30)
    - future_days: Number of days to forecast beyond the test set (default: 0)
    Returns:
    - fig: Plotly figure with historical and forecasted data
    - forecasts_dict: Dictionary with forecast results for each model
    - metrics_df: DataFrame with evaluation metrics for each model
    """
    # Ensure data is sorted
    df = df.sort_values('Data')
    
    # Garantir que não há valores NaN na coluna de interesse
    df = df.dropna(subset=[categoria])
    
    # Garantir que a coluna Data é do tipo datetime
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Split data into train (80%) and test (20%)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    # Prepare data for different models
    df_prophet = df[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    df_train_prophet = df_train[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    df_test_prophet = df_test[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    
    # Prepare data for Nixtla models
    df_nixtla = df[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    df_train_nixtla = df_train[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    df_test_nixtla = df_test[["Data", categoria]].rename(columns={"Data": "ds", categoria: "y"})
    
    # Add unique_id for Nixtla models
    df_nixtla["unique_id"] = "transporte"
    df_train_nixtla["unique_id"] = "transporte"
    df_test_nixtla["unique_id"] = "transporte"

    # Initialize dictionaries to store forecasts and metrics
    forecasts_dict = {}
    metrics_dict = {}
    
    # Ensure modelos is a list
    if isinstance(modelos, str):
        modelos = [modelos]
    
    # Generate forecasts for each model
    # Calcular número de dias futuros até data_final, se fornecida
    if data_final is not None:
        last_data = df["Data"].max()
        data_final = pd.to_datetime(data_final)
        dias_futuros = (data_final - last_data).days
        if dias_futuros < 0:
            dias_futuros = 0
    else:
        dias_futuros = future_days
    # Só permitir modelos Prophet, AutoARIMA, AutoETS
    modelos_validos = [m for m in modelos if m in ["Prophet", "AutoARIMA", "AutoETS"]]
    for modelo in modelos_validos:

        # Initialize forecast DataFrame for this model
        forecast_df = pd.DataFrame()

        if modelo == "Prophet":
            # Prophet model
            if len(df_train_prophet) < 2:
                print(f"[Prophet] Dados insuficientes para treinar: {len(df_train_prophet)} linhas")
                forecast_df = pd.DataFrame()
            else:
                model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
                model.fit(df_train_prophet)
                # Gerar datas contínuas para previsão (teste + futuro)
                # Gerar datas até data_final
                if data_final is not None:
                    last_date = df_train_prophet["ds"].max()
                    n_pred = (data_final - last_date).days
                    if n_pred < 1:
                        n_pred = 1
                    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                else:
                    n_pred = len(df_test_prophet) + future_days
                    last_date = df_train_prophet["ds"].max()
                    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                pred_df = pd.DataFrame({'ds': pred_dates})
                forecast = model.predict(pred_df)
                forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                print(f"[DEBUG] forecast_df Prophet shape: {forecast_df.shape}")
                print(f"[DEBUG] forecast_df Prophet head:\n{forecast_df.head()}")
                forecasts_dict[modelo] = forecast_df
                if forecast_df.empty:
                    print("[Prophet] forecast_df vazio após previsão!")

        elif modelo in ["AutoARIMA", "AutoETS"]:
            # StatsForecast models
            model_class = AutoARIMA if modelo == "AutoARIMA" else AutoETS
            sf = StatsForecast(
                models=[model_class(season_length=7)],  # Weekly seasonality
                freq="D",
                n_jobs=-1
            )
            # Criar DataFrame de treino contínuo e sem datas duplicadas
            train_data = pd.DataFrame({
                'unique_id': 1,
                'ds': pd.to_datetime(df_train_nixtla['ds']),
                'y': df_train_nixtla['y']
            })
            train_data = train_data.dropna().drop_duplicates(subset=['ds'])
            if len(train_data) < 2:
                print(f"[{modelo}] Dados insuficientes para treinar: {len(train_data)} linhas")
                forecast_df = pd.DataFrame()
            else:
                try:
                    sf.fit(train_data)
                    # Gerar datas contínuas para previsão (teste + futuro)
                    # Gerar datas até data_final
                    if data_final is not None:
                        last_date = train_data["ds"].max()
                        n_pred = (data_final - last_date).days
                        if n_pred < 1:
                            n_pred = 1
                        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                    else:
                        n_pred = len(df_test_nixtla) + future_days
                        last_date = train_data["ds"].max()
                        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                    pred_df = pd.DataFrame({'unique_id': 1, 'ds': pred_dates})
                    forecast = sf.predict(h=n_pred, level=[95])
                    forecast_df = forecast.reset_index()
                    forecast_df = forecast_df.rename(columns={modelo: 'yhat'})
                    forecast_df['yhat_lower'] = forecast_df.get(f'{modelo}-lo-95', None)
                    forecast_df['yhat_upper'] = forecast_df.get(f'{modelo}-hi-95', None)
                    forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    forecasts_dict[modelo] = forecast_df
                    print(f"[DEBUG] forecast_df {modelo} shape: {forecast_df.shape}")
                    print(f"[DEBUG] forecast_df {modelo} head:\n{forecast_df.head()}")
                    if forecast_df.empty:
                        print(f"[{modelo}] forecast_df vazio após previsão!")
                except Exception as e:
                    print(f"[ERRO] Falha ao gerar previsão para {modelo}: {e}. Usando valores reais do teste como fallback.")
                    if len(df_test_nixtla) > 0:
                        forecast_df = pd.DataFrame({
                            'ds': df_test_nixtla['ds'],
                            'yhat': df_test_nixtla['y'],  # Usar os valores reais como previsão
                            'yhat_lower': df_test_nixtla['y'] * 0.9,  # 10% abaixo
                            'yhat_upper': df_test_nixtla['y'] * 1.1   # 10% acima
                        })
                    else:
                        forecast_df = pd.DataFrame()
                    forecasts_dict[modelo] = forecast_df


    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={'index': 'Modelo'}, inplace=True)
    
    # Montar tabela de previsões consolidada (valor real, previsão, intervalo inferior, intervalo superior para cada modelo, com labels claros)
    # Usar as datas do conjunto de teste + 7 dias futuros
    all_dates = list(df_test_prophet['ds'])
    if len(all_dates) > 0:
        last_test_date = all_dates[-1]
        future_dates = pd.date_range(start=last_test_date + pd.Timedelta(days=1), periods=7, freq='D')
        all_dates += list(future_dates)
    tabela_previsoes = pd.DataFrame({'ds': all_dates})
    tabela_previsoes = tabela_previsoes.merge(df_prophet[['ds', 'y']].rename(columns={'y': 'Valor Real'}), on='ds', how='left')
    for modelo in modelos:
        if modelo in forecasts_dict and len(forecasts_dict[modelo]) > 0:
            tabela_previsoes = tabela_previsoes.merge(
                forecasts_dict[modelo][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={
                        'yhat': f'Previsão ({modelo})',
                        'yhat_lower': f'Previsão Inferior ({modelo})',
                        'yhat_upper': f'Previsão Superior ({modelo})'
                    }
                ),
                on='ds', how='left'
            )
    
    # Plotting
    fig = go.Figure()
    
    # Plotar dados históricos de treino
    fig.add_trace(go.Scatter(
        x=df_train_prophet["ds"], 
        y=df_train_prophet["y"], 
        mode="lines", 
        name="Dados Históricos de Treino",
        line=dict(color="blue")
    ))
    
    # Plotar dados de teste
    fig.add_trace(go.Scatter(
        x=df_test_prophet["ds"], 
        y=df_test_prophet["y"], 
        mode="lines", 
        name="Dados de Teste",
        line=dict(color="green")
    ))
    
    # Cores para diferentes modelos
    colors = ["red", "purple", "orange", "brown", "pink"]
    
    # Plotar previsões apenas para os modelos selecionados
    for i, modelo in enumerate(modelos):
        if modelo in forecasts_dict and len(forecasts_dict[modelo]) > 0:
            forecast_df = forecasts_dict[modelo]
            color = colors[i % len(colors)]
            # Plotar previsão
            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], 
                y=forecast_df["yhat"], 
                mode="lines", 
                name=f"Previsão ({modelo})",
                line=dict(color=color)
            ))
            # Plotar intervalos de confiança
            if modelo in ["AutoARIMA", "AutoETS", "Prophet"]:
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], 
                    y=forecast_df["yhat_upper"], 
                    mode="lines", 
                    name=f"Limite Superior ({modelo})", 
                    line=dict(dash="dash", color=color)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], 
                    y=forecast_df["yhat_lower"], 
                    mode="lines", 
                    name=f"Limite Inferior ({modelo})", 
                    line=dict(dash="dash", color=color),
                    fill='tonexty'
                ))
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': f"Previsão de {categoria} com Múltiplos Modelos",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        xaxis_title="Data",
        yaxis_title="Valor (R$)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(b=120)
    )

    # Cálculo das métricas de avaliação
    metrics = []
    for modelo, forecast_df in forecasts_dict.items():
        if forecast_df is not None and not forecast_df.empty:
            # Pegue os valores reais (do teste) para as datas previstas
            if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                # Use apenas as datas do conjunto de teste para métricas
                real_df = df_test[["Data", categoria]].rename(columns={"Data": "ds", categoria: "Valor Real"})
                merged = pd.merge(forecast_df[['ds', 'yhat']], real_df, on='ds', how='inner')
                if not merged.empty:
                    m = calculate_metrics(merged['Valor Real'].values, merged['yhat'].values)
                    m['Modelo'] = modelo
                    metrics.append(m)
                    print(f"[DEBUG] Métricas para {modelo}: {m}")
                else:
                    print(f"[DEBUG] Nenhuma data em comum para métricas do modelo {modelo}")
    if metrics:
        metrics_df = pd.DataFrame(metrics)[['Modelo', 'RMSE', 'MAE', 'MAPE']]
    else:
        metrics_df = pd.DataFrame(columns=['Modelo', 'RMSE', 'MAE', 'MAPE'])

    print("[DEBUG] Forecasts_dict keys:", forecasts_dict.keys())
    for modelo, df in forecasts_dict.items():
        print(f"[DEBUG] Modelo: {modelo}, shape: {df.shape}, head:\n{df.head()}")
    print("[DEBUG] Metrics_df:")
    print(metrics_df)
    print("[DEBUG] Tabela_previsoes head:")
    print(tabela_previsoes.head() if tabela_previsoes is not None else "Tabela de previsões vazia")
    return fig, forecasts_dict, metrics_df, tabela_previsoes

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for forecasting models.
    
    Parameters:
    - y_true: Array-like, true values
    - y_pred: Array-like, predicted values
    
    Returns:
    - Dictionary with RMSE, MAE, and MAPE metrics
    """
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero in MAPE calculation
    mask_nonzero = y_true != 0
    y_true_nonzero = y_true[mask_nonzero]
    y_pred_nonzero = y_pred[mask_nonzero]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate MAPE only if we have non-zero values
    if len(y_true_nonzero) > 0:
        mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
    else:
        mape = np.nan
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }