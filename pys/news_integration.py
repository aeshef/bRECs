import pandas as pd
import numpy as np
import os

class NewsIntegration:
    """Класс для интеграции новостных данных с моделями прогнозирования"""
    
    def __init__(self):
        """Инициализация интегратора новостей"""
        pass
    
    def merge_news_with_price_data(self, news_features_df, price_df, date_column='date'):
        """
        Объединение новостных признаков с ценовыми данными
        
        Args:
            news_features_df (pd.DataFrame): DataFrame с признаками новостей
            price_df (pd.DataFrame): DataFrame с ценовыми данными
            date_column (str): Название колонки с датой
            
        Returns:
            pd.DataFrame: Объединенный DataFrame
        """
        if news_features_df.empty or price_df.empty:
            print("Один из входных DataFrame пуст")
            return pd.DataFrame()
        

        news_features_df[date_column] = pd.to_datetime(news_features_df[date_column])
        
        if isinstance(price_df.index, pd.DatetimeIndex):
            price_df_copy = price_df.copy()
        else:
            date_col = date_column if date_column in price_df.columns else price_df.columns[0]
            price_df_copy = price_df.copy()
            price_df_copy[date_col] = pd.to_datetime(price_df_copy[date_col])
            price_df_copy.set_index(date_col, inplace=True)
        
        news_features_index = news_features_df.set_index(date_column)
        
        combined_df = price_df_copy.join(news_features_index, how='left')
        
        numeric_cols = news_features_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in combined_df.columns and col != date_column:
                combined_df[col].fillna(0, inplace=True)
        
        categorical_cols = news_features_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in combined_df.columns and col != date_column:
                most_common = news_features_df[col].mode()[0]
                combined_df[col].fillna(most_common, inplace=True)
        
        return combined_df
    
    def create_ml_features(self, combined_df, target_column='close', prediction_horizon=5):
        """
        Создание признаков для моделей ML на основе объединенных данных
        
        Args:
            combined_df (pd.DataFrame): Объединенный DataFrame с ценами и новостными признаками
            target_column (str): Название колонки с целевой переменной (обычно цена закрытия)
            prediction_horizon (int): Горизонт прогнозирования (в днях)
            
        Returns:
            pd.DataFrame: DataFrame с признаками для ML
        """
        if combined_df.empty or target_column not in combined_df.columns:
            print(f"DataFrame пуст или не содержит колонку {target_column}")
            return pd.DataFrame()
        
        ml_df = combined_df.copy()
        ml_df['target_return'] = ml_df[target_column].pct_change(prediction_horizon).shift(-prediction_horizon)
        news_features = [col for col in ml_df.columns if col.startswith(('sentiment', 'news_count', 'event_', 'topic_'))]

        for feature in news_features:
            for lag in [1, 2, 3, 5, 10]:
                ml_df[f'{feature}_lag{lag}'] = ml_df[feature].shift(lag)
        
        for feature in news_features:
            for window in [3, 7, 14, 30]:
                ml_df[f'{feature}_ma{window}'] = ml_df[feature].rolling(window=window, min_periods=1).mean()
            
            ml_df[f'{feature}_std7'] = ml_df[feature].rolling(window=7, min_periods=1).std()
            ml_df[f'{feature}_std14'] = ml_df[feature].rolling(window=14, min_periods=1).std()
        
        if 'avg_sentiment' in ml_df.columns:
            ml_df['sentiment_return_corr'] = ml_df['avg_sentiment'].rolling(window=14).corr(ml_df[target_column].pct_change())
        
        if 'news_count' in ml_df.columns:
            ml_df['news_count_volatility'] = ml_df['news_count'] * ml_df[target_column].pct_change().rolling(window=7).std()
        
        return ml_df
    
    def convert_news_to_views(self, news_features_df, market_returns, risk_aversion=2.5):
        """
        Конвертация новостных сигналов в мнения инвесторов для модели Блека-Литермана
        
        Args:
            news_features_df (pd.DataFrame): DataFrame с признаками новостей
            market_returns (dict): Словарь {ticker: expected_return} с историческими доходностями
            risk_aversion (float): Коэффициент неприятия риска
            
        Returns:
            dict: Словарь {ticker: (expected_return, confidence)} с мнениями инвесторов
        """
        if news_features_df.empty or 'ticker' not in news_features_df.columns:
            print("DataFrame пуст или не содержит колонку 'ticker'")
            return {}
        
        latest_date = news_features_df['date'].max()
        latest_data = news_features_df[news_features_df['date'] == latest_date]
        
        views = {}
        
        for _, row in latest_data.iterrows():
            ticker = row['ticker']
            
            base_return = market_returns.get(ticker, 0)
            
            sentiment_adjustment = 0
            if 'sentiment_compound' in row:
                sentiment_adjustment = row['sentiment_compound'] * 0.01
            elif 'avg_sentiment' in row:
                sentiment_adjustment = row['avg_sentiment'] * 0.01

            news_volume_factor = 0
            if 'news_count' in row:
                news_volume_factor = min(row['news_count'] / 10, 1.0)
            
            event_importance = 0
            for col in row.index:
                if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']:
                    event_importance += row[col] if isinstance(row[col], (int, float)) else 0
            
            event_factor = min(event_importance * 0.02, 0.1)
            adjusted_return = base_return + sentiment_adjustment + event_factor
            confidence = (0.5 + news_volume_factor * 0.3 + event_importance * 0.2) / risk_aversion
            views[ticker] = (adjusted_return, confidence)
        
        return views
