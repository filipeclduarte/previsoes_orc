import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    # Permitir modelos Prophet, AutoARIMA, AutoETS, MLP, SVR
    modelos_validos = [m for m in modelos if m in ["Prophet", "AutoARIMA", "AutoETS", "MLP", "SVR"]]
    for modelo in modelos_validos:

        # Initialize forecast DataFrame for this model
        forecast_df = pd.DataFrame()

        if modelo == "Prophet":            # Prophet model
            y_pred_test = np.array([])
            try:
                if len(df_train_prophet) < 2:
                    print(f"[Prophet] Dados insuficientes para treinar: {len(df_train_prophet)} linhas")
                    forecast_df = pd.DataFrame()
                else:
                    model = Prophet()
                    model.fit(df_train_prophet)
                    # Previsão para teste + futuros
                    future_dates = df_test_prophet[['ds']].copy()
                    if dias_futuros > 0:
                        last_date = df["Data"].max()
                        future_extra = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=dias_futuros)
                        future_dates = pd.concat([future_dates, pd.DataFrame({'ds': future_extra})], ignore_index=True)
                    forecast = model.predict(future_dates)
                    forecast_df = pd.DataFrame({
                        'ds': forecast['ds'],
                        'yhat': forecast['yhat'],
                        'yhat_lower': forecast['yhat_lower'],
                        'yhat_upper': forecast['yhat_upper']
                    })
                    y_pred_test = forecast_df.loc[forecast_df['ds'].isin(df_test_prophet['ds']), 'yhat'].values
                    forecasts_dict[modelo] = forecast_df
            except Exception as e:
                print(f"[ERRO] Falha na previsão recursiva para {modelo}: {e}")
                forecasts_dict[modelo] = pd.DataFrame()

        elif modelo == "MLP":
            y_pred_test = np.array([])
            # MLPRegressor com GridSearchCV
            X_train = np.arange(len(df_train)).reshape(-1, 1)
            y_train = df_train[categoria].values
            X_test = np.arange(len(df_train), len(df_train) + len(df_test)).reshape(-1, 1)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(max_iter=2000, random_state=42))
            ])
            param_grid = {
                "mlp__hidden_layer_sizes": [(5,), (10,), (15,), (25,), (50,), (5, 5), (10, 10), (15, 15)],
                "mlp__activation": ["relu", "tanh"],
                "mlp__alpha": [0.0001, 0.001, 0.01]
            }
            gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"[DEBUG] Melhor params MLP: {gs.best_params_}")
            # Previsão para teste
            y_pred_test = gs.predict(X_test)
            forecast_df = pd.DataFrame({
                "ds": df_test["Data"].values,
                "yhat": y_pred_test
            })
            # Previsão recursiva até data_final
            try:
                # Conformal prediction para intervalos de previsão (split conformal)
                if len(df_test) > 0 and len(y_pred_test) == len(df_test[categoria]):
                    residuos_abs = np.abs(df_test[categoria].values - y_pred_test)
                    quantil = np.quantile(residuos_abs, 0.95)
                else:
                    quantil = np.nan
                # Para cada previsão (teste e futuro)
                forecast_df['yhat_lower'] = forecast_df['yhat'] - quantil
                forecast_df['yhat_upper'] = forecast_df['yhat'] + quantil
                if dias_futuros > 0:
                    n_start = len(df_train) + len(df_test)
                    n_end = n_start + dias_futuros
                    if n_end > n_start:
                        X_future = np.arange(n_start, n_end).reshape(-1, 1)
                        y_pred_future = gs.predict(X_future)
                        datas_futuras = pd.date_range(start=df["Data"].max() + pd.Timedelta(days=1), periods=dias_futuros)
                        # Intervalo para previsão futura (usar resíduos do teste)
                        # lower_future, upper_future = gerar_intervalo(y_pred_future)
                        # Simulação recursiva com bootstrapped residuals para intervalos futuros
                        n_boot = 1000
                        rng = np.random.default_rng(42)
                        # Calcular resíduos do histórico (treino + validação)
                        X_all = np.concatenate([X_train, X_test]) if len(X_test) > 0 else X_train
                        y_all = np.concatenate([y_train, df_test[categoria].values]) if len(df_test) > 0 else y_train
                        y_pred_all = gs.predict(X_all)
                        residuos = y_all - y_pred_all if len(y_all) == len(y_pred_all) else y_train - gs.predict(X_train)
                        # Previsão média (ponto) já está calculada em y_pred_future
                        # Simular caminhos futuros
                        sim_paths = np.zeros((n_boot, len(y_pred_future)))
                        for b in range(n_boot):
                            y_sim = []
                            last_value = df["Valor Real"].values[-1] if "Valor Real" in df.columns else df[categoria].values[-1]
                            for i in range(len(y_pred_future)):
                                # Para modelos autoregressivos, aqui deveria atualizar os lags
                                pred = y_pred_future[i]
                                erro = rng.choice(residuos)
                                y_next = pred + erro
                                y_sim.append(y_next)
                            sim_paths[b, :] = y_sim
                        lower_future = np.percentile(sim_paths, 2.5, axis=0)
                        upper_future = np.percentile(sim_paths, 97.5, axis=0)
                        # Garantir alinhamento de tamanho
                        min_len = min(len(datas_futuras), len(y_pred_future), len(lower_future), len(upper_future))
                        forecast_future = pd.DataFrame({
                            "ds": datas_futuras[:min_len],
                            "yhat": y_pred_future[:min_len],
                            "yhat_lower": lower_future[:min_len],
                            "yhat_upper": upper_future[:min_len]
                        })
                        forecast_df = pd.concat([forecast_df, forecast_future], ignore_index=True)
                        # Para o teste, pode-se usar conformal prediction ou o mesmo método se desejar
                        # Aqui, para simplificar, mantemos os intervalos do futuro apenas para as datas futuras
                        forecast_df.loc[forecast_df['ds'].isin(datas_futuras), 'yhat_lower'] = lower_future
                        forecast_df.loc[forecast_df['ds'].isin(datas_futuras), 'yhat_upper'] = upper_future
                    else:
                        forecast_df['yhat_lower'] = np.nan
                        forecast_df['yhat_upper'] = np.nan
                else:
                    forecast_df['yhat_lower'] = np.nan
                    forecast_df['yhat_upper'] = np.nan
                forecasts_dict[modelo] = forecast_df
            except Exception as e:
                print(f"[ERRO] Falha na previsão recursiva para {modelo}: {e}")
                forecasts_dict[modelo] = pd.DataFrame()
        elif modelo == "SVR":
            y_pred_test = np.array([])
            # SVR com GridSearchCV
            X_train = np.arange(len(df_train)).reshape(-1, 1)
            y_train = df_train[categoria].values
            X_test = np.arange(len(df_train), len(df_train) + len(df_test)).reshape(-1, 1)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR())
            ])
            param_grid = {
                "svr__C": [0.1, 1, 10],
                "svr__gamma": ["scale", "auto"],
                "svr__kernel": ["rbf", "linear", "poly", "sigmoid"]
            }
            gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"[DEBUG] Melhor params SVR: {gs.best_params_}")
            # Previsão para teste
            y_pred_test = gs.predict(X_test)
            forecast_df = pd.DataFrame({
                "ds": df_test["Data"].values,
                "yhat": y_pred_test
            })
            # Previsão recursiva até data_final
            try:
                # Conformal prediction para intervalos de previsão (split conformal)
                if len(df_test) > 0 and len(y_pred_test) == len(df_test[categoria]):
                    residuos_abs = np.abs(df_test[categoria].values - y_pred_test)
                    quantil = np.quantile(residuos_abs, 0.95)
                else:
                    quantil = np.nan
                # Para cada previsão (teste e futuro)
                forecast_df['yhat_lower'] = forecast_df['yhat'] - quantil
                forecast_df['yhat_upper'] = forecast_df['yhat'] + quantil
                if dias_futuros > 0:
                    n_start = len(df_train) + len(df_test)
                    n_end = n_start + dias_futuros
                    if n_end > n_start:
                        X_future = np.arange(n_start, n_end).reshape(-1, 1)
                        y_pred_future = gs.predict(X_future)
                        datas_futuras = pd.date_range(start=df["Data"].max() + pd.Timedelta(days=1), periods=dias_futuros)
                        # Intervalo para previsão futura (usar resíduos do teste)
                        # lower_future, upper_future = gerar_intervalo(y_pred_future)
                        # Simulação recursiva com bootstrapped residuals para intervalos futuros
                        n_boot = 1000
                        rng = np.random.default_rng(42)
                        # Calcular resíduos do histórico (treino + validação)
                        X_all = np.concatenate([X_train, X_test]) if len(X_test) > 0 else X_train
                        y_all = np.concatenate([y_train, df_test[categoria].values]) if len(df_test) > 0 else y_train
                        y_pred_all = gs.predict(X_all)
                        residuos = y_all - y_pred_all if len(y_all) == len(y_pred_all) else y_train - gs.predict(X_train)
                        # Previsão média (ponto) já está calculada em y_pred_future
                        # Simular caminhos futuros
                        sim_paths = np.zeros((n_boot, len(y_pred_future)))
                        for b in range(n_boot):
                            y_sim = []
                            last_value = df["Valor Real"].values[-1] if "Valor Real" in df.columns else df[categoria].values[-1]
                            for i in range(len(y_pred_future)):
                                # Para modelos autoregressivos, aqui deveria atualizar os lags
                                pred = y_pred_future[i]
                                erro = rng.choice(residuos)
                                y_next = pred + erro
                                y_sim.append(y_next)
                            sim_paths[b, :] = y_sim
                        lower_future = np.percentile(sim_paths, 2.5, axis=0)
                        upper_future = np.percentile(sim_paths, 97.5, axis=0)
                        # Garantir alinhamento de tamanho
                        min_len = min(len(datas_futuras), len(y_pred_future), len(lower_future), len(upper_future))
                        forecast_future = pd.DataFrame({
                            "ds": datas_futuras[:min_len],
                            "yhat": y_pred_future[:min_len],
                            "yhat_lower": lower_future[:min_len],
                            "yhat_upper": upper_future[:min_len]
                        })
                        forecast_df = pd.concat([forecast_df, forecast_future], ignore_index=True)
                        # Para o teste, pode-se usar conformal prediction ou o mesmo método se desejar
                        # Aqui, para simplificar, mantemos os intervalos do futuro apenas para as datas futuras
                        forecast_df.loc[forecast_df['ds'].isin(datas_futuras), 'yhat_lower'] = lower_future
                        forecast_df.loc[forecast_df['ds'].isin(datas_futuras), 'yhat_upper'] = upper_future
                    else:
                        forecast_df['yhat_lower'] = np.nan
                        forecast_df['yhat_upper'] = np.nan
                else:
                    forecast_df['yhat_lower'] = np.nan
                    forecast_df['yhat_upper'] = np.nan
                forecasts_dict[modelo] = forecast_df
            except Exception as e:
                print(f"[ERRO] Falha na previsão recursiva para {modelo}: {e}")
                forecasts_dict[modelo] = pd.DataFrame()

        # O bloco else: abaixo foi removido pois não é necessário após SVR/MLP

        elif modelo == "AutoARIMA":
            print(f"[DEBUG] Entrando no bloco AutoARIMA")
            y_pred_test = np.array([])
            try:
                # Preparar dados para StatsForecast
                train_data = pd.DataFrame({
                    'unique_id': 1,
                    'ds': pd.to_datetime(df_train_nixtla['ds']),
                    'y': df_train_nixtla['y']
                })
                train_data = train_data.dropna().drop_duplicates(subset=['ds'])
                if len(train_data) < 2:
                    print(f"[AutoARIMA] Dados insuficientes para treinar: {len(train_data)} linhas")
                    forecast_df = pd.DataFrame()
                else:
                    sf = StatsForecast(
                        models=[AutoARIMA(season_length=7, max_p=7, max_q=7)],
                        freq="D",
                        n_jobs=-1
                    )
                    sf.fit(train_data)
                    data_final = pd.to_datetime("2025-04-24")
                    last_date = train_data["ds"].max()
                    n_pred = (data_final - last_date).days
                    if n_pred < 1:
                        n_pred = 1
                    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                    pred_df = pd.DataFrame({'unique_id': 1, 'ds': pred_dates})
                    forecast = sf.predict(h=n_pred, level=[95])
                    print(f"[DEBUG] forecast columns for AutoARIMA: {forecast.columns.tolist()}")
                    print(f"[DEBUG] forecast head for AutoARIMA:\n{forecast.head()}")
                    forecast_df = forecast.reset_index(drop=True)
                    # Extrair coluna de previsão
                    yhat_col = [col for col in forecast_df.columns if col.lower() in ['autoarima', 'yhat', 'forecast']]
                    forecast_df['yhat'] = forecast_df[yhat_col[0]] if yhat_col else np.nan
                    lo_col = [col for col in forecast_df.columns if col.endswith('-lo-95')]
                    hi_col = [col for col in forecast_df.columns if col.endswith('-hi-95')]
                    forecast_df['yhat_lower'] = forecast_df[lo_col[0]] if lo_col else np.nan
                    forecast_df['yhat_upper'] = forecast_df[hi_col[0]] if hi_col else np.nan
                    forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    print(f"[DEBUG] forecast_df AutoARIMA shape: {forecast_df.shape}")
                    print(f"[DEBUG] forecast_df AutoARIMA head:\n{forecast_df.head()}")
                    print(f"[DEBUG] forecast_df AutoARIMA columns: {forecast_df.columns.tolist()}")
                    if forecast_df.empty:
                        print(f"[AutoARIMA] forecast_df vazio após previsão!")
                    print(f"[DEBUG] Saindo do bloco AutoARIMA")
            except Exception as e:
                print(f"[ERRO][EXCEPTION] Falha ao gerar previsão para AutoARIMA: {e}. type={type(e)}")
                import traceback; traceback.print_exc()
                forecast_df = pd.DataFrame()
            forecasts_dict[modelo] = forecast_df

        elif modelo == "AutoETS":
            print(f"[DEBUG] Entrando no bloco AutoETS")
            y_pred_test = np.array([])
            try:
                train_data = pd.DataFrame({
                    'unique_id': 1,
                    'ds': pd.to_datetime(df_train_nixtla['ds']),
                    'y': df_train_nixtla['y']
                })
                train_data = train_data.dropna().drop_duplicates(subset=['ds'])
                if len(train_data) < 2:
                    print(f"[AutoETS] Dados insuficientes para treinar: {len(train_data)} linhas")
                    forecast_df = pd.DataFrame()
                else:
                    sf = StatsForecast(
                        models=[AutoETS(season_length=7)],
                        freq="D",
                        n_jobs=-1
                    )
                    sf.fit(train_data)
                    data_final = pd.to_datetime("2025-04-24")
                    last_date = train_data["ds"].max()
                    n_pred = (data_final - last_date).days
                    if n_pred < 1:
                        n_pred = 1
                    pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_pred, freq='D')
                    pred_df = pd.DataFrame({'unique_id': 1, 'ds': pred_dates})
                    forecast = sf.predict(h=n_pred, level=[95])
                    print(f"[DEBUG] forecast columns for AutoETS: {forecast.columns.tolist()}")
                    print(f"[DEBUG] forecast head for AutoETS:\n{forecast.head()}")
                    forecast_df = forecast.reset_index(drop=True)
                    yhat_col = [col for col in forecast_df.columns if col.lower() in ['autoets', 'yhat', 'forecast']]
                    forecast_df['yhat'] = forecast_df[yhat_col[0]] if yhat_col else np.nan
                    lo_col = [col for col in forecast_df.columns if col.endswith('-lo-95')]
                    hi_col = [col for col in forecast_df.columns if col.endswith('-hi-95')]
                    forecast_df['yhat_lower'] = forecast_df[lo_col[0]] if lo_col else np.nan
                    forecast_df['yhat_upper'] = forecast_df[hi_col[0]] if hi_col else np.nan
                    forecast_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    print(f"[DEBUG] forecast_df AutoETS shape: {forecast_df.shape}")
                    print(f"[DEBUG] forecast_df AutoETS head:\n{forecast_df.head()}")
                    print(f"[DEBUG] forecast_df AutoETS columns: {forecast_df.columns.tolist()}")
                    if forecast_df.empty:
                        print(f"[AutoETS] forecast_df vazio após previsão!")
                    print(f"[DEBUG] Saindo do bloco AutoETS")
            except Exception as e:
                print(f"[ERRO][EXCEPTION] Falha ao gerar previsão para AutoETS: {e}. type={type(e)}")
                import traceback; traceback.print_exc()
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
    
    # --- PÓS-PROCESSAMENTO: ZERAR PREVISÕES NEGATIVAS NOS DATAFRAMES DE forecasts_dict ---
    for modelo, forecast_df in forecasts_dict.items():
        if forecast_df is not None and not forecast_df.empty:
            for col in ['yhat', 'yhat_lower']:
                if col in forecast_df.columns:
                    forecast_df[col] = np.maximum(0, forecast_df[col])

    # --- PÓS-PROCESSAMENTO: ZERAR NEGATIVOS NA TABELA DE PREVISÕES ---
    for col in tabela_previsoes.columns:
        if col.startswith('Previsão (') or col.startswith('Previsão Inferior ('):
            tabela_previsoes[col] = np.maximum(0, tabela_previsoes[col])

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
            if modelo in ["AutoARIMA", "AutoETS", "Prophet", "MLP", "SVR"]:
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], 
                    y=forecast_df["yhat_upper"], 
                    mode="lines", 
                    name=f"Limite Superior ({modelo})", 
                    line=dict(dash="dash", color=color)
                ))
                # Definir cor transparente para o preenchimento
                fill_rgba = color
                if color == "red":
                    fill_rgba = "rgba(255,0,0,0.15)"
                elif color == "purple":
                    fill_rgba = "rgba(128,0,128,0.15)"
                elif color == "orange":
                    fill_rgba = "rgba(255,140,0,0.15)"
                elif color == "brown":
                    fill_rgba = "rgba(139,69,19,0.15)"
                elif color == "pink":
                    fill_rgba = "rgba(255,105,180,0.15)"
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], 
                    y=forecast_df["yhat_lower"], 
                    mode="lines", 
                    name=f"Limite Inferior ({modelo})", 
                    line=dict(dash="dash", color=color),
                    fill='tonexty',
                    fillcolor=fill_rgba
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

    print(f"[DEBUG][GLOBAL] forecasts_dict gerado:")
    for modelo, forecast_df in forecasts_dict.items():
        print(f"  Modelo: {modelo}, forecast_df shape: {forecast_df.shape if forecast_df is not None else None}")
        if forecast_df is not None:
            print(forecast_df.head())

    # --- PÓS-PROCESSAMENTO: ZERAR PREVISÕES NEGATIVAS ---
    for modelo, forecast_df in forecasts_dict.items():
        if forecast_df is not None and not forecast_df.empty:
            for col in ['yhat', 'yhat_lower']:
                if col in forecast_df.columns:
                    forecast_df[col] = np.maximum(0, forecast_df[col])

    # Calcular métricas para cada modelo
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

    print("[DEBUG] Metrics_df:")
    print(metrics_df)
    print("[DEBUG] Tabela_previsoes head:")
    print(tabela_previsoes.head() if tabela_previsoes is not None else "Tabela de previsões vazia")

    print(f"[DEBUG][GLOBAL] metrics_df calculado:\n{metrics_df}")
    if metrics_df.empty:
        print("[ERRO][GLOBAL] metrics_df está vazio! Nenhuma previsão foi considerada válida para cálculo de métricas.")
    # Selecionar melhor modelo
    melhor_modelo, metrics_df_ordenado = selecionar_melhor_modelo(metrics_df)
    print(f"[DEBUG] Melhor modelo: {melhor_modelo}")
    return fig, forecasts_dict, metrics_df, tabela_previsoes, melhor_modelo, metrics_df_ordenado

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


def selecionar_melhor_modelo(metrics_df):
    """
    Seleciona o melhor modelo com base nos menores valores normalizados de RMSE, MAE e MAPE.
    Retorna o nome do melhor modelo e o DataFrame ordenado.
    """
    if metrics_df.empty:
        return None, metrics_df
    for metrica in ['RMSE', 'MAE', 'MAPE']:
        metrics_df[metrica + '_norm'] = metrics_df[metrica] / metrics_df[metrica].min()
    metrics_df['score_total'] = metrics_df[['RMSE_norm', 'MAE_norm', 'MAPE_norm']].sum(axis=1)
    melhor_modelo = metrics_df.loc[metrics_df['score_total'].idxmin(), 'Modelo']
    return melhor_modelo, metrics_df.sort_values('score_total')