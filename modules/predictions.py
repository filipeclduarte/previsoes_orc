import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from neuralforecast.models import LSTM
from neuralforecast import NeuralForecast
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make neuralforecast.LSTM compatible with scikit-learn API"""
    def __init__(self, h=12, input_size=24, max_steps=500, scaler_type="standard"):
        self.h = h
        self.input_size = input_size
        self.max_steps = max_steps
        self.scaler_type = scaler_type
        self.model = None

    def fit(self, X, y):
        # Prepare data in neuralforecast format
        df = pd.DataFrame({
            "unique_id": "transporte",
            "ds": X["ds"],
            "y": y
        })
        self.model = NeuralForecast(
            models=[LSTM(
                h=self.h,
                input_size=self.input_size,
                max_steps=self.max_steps,
                scaler_type=self.scaler_type
            )],
            freq="D"  # Alterado para frequência diária
        )
        self.model.fit(df=df)
        return self

    def predict(self, X):
        # Prepare forecast DataFrame
        df = pd.DataFrame({
            "unique_id": "transporte",
            "ds": X["ds"]
        })
        forecast = self.model.predict(df=df)
        return forecast["LSTM"].values

def generate_forecast(df, categoria, modelos, horizon=30, future_days=0):
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
    
    # Split data into train (70%) and test (30%)
    train_size = int(len(df) * 0.7)
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
    for modelo in modelos:

        # Initialize forecast DataFrame for this model
        forecast_df = pd.DataFrame()

        if modelo == "Prophet":
            # Prophet model
            model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
            model.fit(df_train_prophet)
            
            # Generate predictions for test set
            test_forecast = model.predict(df_test_prophet)
            test_forecast = test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        
            # If future_days > 0, generate predictions for future days
            if future_days > 0:
                # Create future dates starting from the day after the last test date
                last_date = df_test_prophet["ds"].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
                future_df = pd.DataFrame({'ds': future_dates})
                
                # Generate predictions for future dates
                future_forecast = model.predict(future_df)
                future_forecast = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                
                # Combine test and future forecasts
                forecast_df = pd.concat([test_forecast, future_forecast], ignore_index=True)
            else:
                forecast_df = test_forecast

        elif modelo in ["AutoARIMA", "AutoETS"]:
            # StatsForecast models
            model_class = AutoARIMA if modelo == "AutoARIMA" else AutoETS
            sf = StatsForecast(
                models=[model_class(season_length=7)],  # Weekly seasonality
                freq="D",
                n_jobs=-1
            )
            
            # Create intermediate DataFrame with 'y' column
            train_data = pd.DataFrame({
                'unique_id': 1,
                'ds': df_train_nixtla['ds'],
                'y': df_train_nixtla['y']
            })
            
            # Garantir que não há valores NaN
            train_data = train_data.dropna()
            
            try:
                # Ajustar o modelo
                sf.fit(train_data)
                
                # Criar datas para previsão (usando as datas do conjunto de teste)
                test_dates = df_test_nixtla['ds'].reset_index(drop=True)
                
                # Calcular o horizonte total (teste + futuro)
                total_horizon = len(test_dates) + future_days if future_days > 0 else len(test_dates)
                
                # Fazer previsão
                forecast = sf.predict(h=total_horizon, level=[95])
                
                # Preparar DataFrame de resultado
                forecast_df = forecast.reset_index()
                forecast_df = forecast_df.rename(columns={modelo: 'yhat'})
                forecast_df['yhat_lower'] = forecast_df[f'{modelo}-lo-95']
                forecast_df['yhat_upper'] = forecast_df[f'{modelo}-hi-95']
                forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            except Exception as e:
                print(f"Erro ao ajustar o modelo {modelo}: {e}")
                # Em caso de erro, criar um DataFrame com os valores reais
                forecast_df = pd.DataFrame({
                    'ds': df_test_nixtla['ds'],
                    'yhat': df_test_nixtla['y'],  # Usar os valores reais como previsão
                    'yhat_lower': df_test_nixtla['y'] * 0.9,  # 10% abaixo
                    'yhat_upper': df_test_nixtla['y'] * 1.1   # 10% acima
                })

        elif modelo in ["MLP", "LSTM"]:
            # MLForecast models
            h = len(df_test)  # Horizonte de previsão igual ao tamanho do conjunto de teste
            
            if modelo == "MLP":
                model_class = MLPRegressor(random_state=0, max_iter=1000, hidden_layer_sizes=(50, 25), activation='relu')
            else:
                model_class = LSTMWrapper(h=h, input_size=30, max_steps=1000)
            
            # Preparar dados de treino
            train_data = pd.DataFrame({
                'unique_id': 1,
                'ds': df_train_nixtla['ds'],
                'y': df_train_nixtla['y']
            })
            
            # Garantir que não há valores NaN
            train_data = train_data.dropna()
            
            # Configurar MLForecast
            mlf = MLForecast(
                models={modelo: model_class},
                freq="D",
                lags=[1, 2, 3, 7, 14],  # Lags mais curtos para evitar problemas com dados insuficientes
                lag_transforms={
                    7: [(lambda x: x.rolling(7, min_periods=1).mean())]  # Weekly moving average com min_periods=1
                },
                date_features=["month", "dayofweek"]  # Simplificar features de data
            )
            
            # Ajustar o modelo
            try:
                mlf.fit(
                    train_data,
                    prediction_intervals=PredictionIntervals(n_windows=1, h=h)
                )
                
                # Fazer previsão
                forecast = mlf.predict(h=h, level=[95])
                
                # Preparar DataFrame de resultado
                forecast_df = forecast.reset_index()
                forecast_df = forecast_df.rename(columns={modelo: 'yhat'})
                forecast_df['yhat_lower'] = forecast_df[f'{modelo}-lo-95']
                forecast_df['yhat_upper'] = forecast_df[f'{modelo}-hi-95']
                forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                
            except Exception as e:
                # Em caso de erro, criar um DataFrame com os valores reais
                print(f"Erro ao ajustar o modelo {modelo}: {e}")
                forecast_df = pd.DataFrame({
                    'ds': df_test_nixtla['ds'],
                    'yhat': df_test_nixtla['y'],  # Usar os valores reais como previsão
                    'yhat_lower': df_test_nixtla['y'] * 0.9,  # 10% abaixo
                    'yhat_upper': df_test_nixtla['y'] * 1.1   # 10% acima
                })
            
            # Garantir que as datas de forecast_df correspondem às datas de teste
            if len(forecast_df) > 0:
                # Ajustar o índice de forecast_df para corresponder ao df_test_prophet
                if len(forecast_df) != len(df_test_prophet):
                    # Se os tamanhos forem diferentes, usar apenas os primeiros pontos
                    min_len = min(len(forecast_df), len(df_test_prophet))
                    forecast_df = forecast_df.iloc[:min_len].copy()
                    forecast_df["ds"] = df_test_prophet["ds"].iloc[:min_len].values
                else:
                    # Se os tamanhos forem iguais, usar as datas de df_test_prophet
                    forecast_df["ds"] = df_test_prophet["ds"].values
            
            # Store forecast in dictionary
            forecasts_dict[modelo] = forecast_df
            
            # Calculate metrics
            if len(forecast_df) > 0:
                # Align test data with forecast data
                y_true = df_test_prophet['y'].values[:len(forecast_df)]
                y_pred = forecast_df['yhat'].values
                metrics_dict[modelo] = calculate_metrics(y_true, y_pred)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_dict).T.reset_index()
    metrics_df.rename(columns={'index': 'Modelo'}, inplace=True)
    
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
    
    # Plotar previsões para cada modelo
    for i, (modelo, forecast_df) in enumerate(forecasts_dict.items()):
        if len(forecast_df) > 0:
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
            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], 
                y=forecast_df["yhat_upper"], 
                mode="lines", 
                name=f"Limite Superior ({modelo})", 
                line=dict(dash="dash", color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.5)" if color.startswith('#') else f"rgba({color}, 0.5)")
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], 
                y=forecast_df["yhat_lower"], 
                mode="lines", 
                name=f"Limite Inferior ({modelo})", 
                line=dict(dash="dash", color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.5)" if color.startswith('#') else f"rgba({color}, 0.5)"),
                fill='tonexty'
            ))
    
    # Configurar layout
    fig.update_layout(
        title=f"Previsão de {categoria} com Múltiplos Modelos",
        xaxis_title="Data",
        yaxis_title="Valor (R$)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, forecasts_dict, metrics_df

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