import pandas as pd
import numpy as np

class DataIntegrator:
    def __init__(self, base_path, nan_fill_method='median'):
        """
        Параметры:
            base_path - базовый путь до папки с данными, в которой находятся подпапки:
                        AFKS/news_analysis и AFKS/tech_analysis
        """
        self.base_path = base_path
        self.nan_fill_method = nan_fill_method


    def load_ticker_data(self, ticker):
        """
        Загружает и интегрирует данные для одного тикера.
        
        Параметры:
            ticker - тикер актива (например, "SBER")
            
        Возвращает:
            df - объединённый DataFrame с данными для данного тикера, содержащий колонку 'ticker'
        """
        ml_path = f'{self.base_path}/{ticker}/news_analysis/{ticker}_ml_features.csv'
        sentiment_path = f'{self.base_path}/{ticker}/news_analysis/{ticker}_sentiment_features.csv'
        tech_path = f'{self.base_path}/{ticker}/tech_analysis/{ticker}_tech_indicators.csv'

        ml_features = pd.read_csv(ml_path)
        sentiment_features = pd.read_csv(sentiment_path)
        base_with_tech = pd.read_csv(tech_path)

        df = pd.merge(
            pd.merge(base_with_tech, sentiment_features, how='left', on='date'),
            ml_features, how='left', on='date'
        )

        cols_to_drop = [
            'Unnamed: 0',
            'open_y',
            'close_y',
            'high',
            'low',
            'volume_y',
            'avg_sentiment_y',
            'news_count_y'
        ]
        df = df.loc[:, ~df.columns.isin(cols_to_drop)]

        df = df.rename(columns={
            "open_x": "open",
            "close_x": "close",
            "volume_x": "volume"
        })

        df['ticker'] = ticker

        return df
    
    def _fill_nan_in_columns(self, df):
        """
        Заполняет пропуски для указанных колонок (если заданы) или
        для всех числовых колонок, исключая базовые: date, open, close, volume, ticker.
        Метод заполнения определяется параметром nan_fill_method.
        """
        base_cols = ['date', 'open', 'close', 'volume', 'ticker']
        num_cols = df.select_dtypes(include=[np.number]).columns.difference(base_cols)

        for col in num_cols:
            if df[col].isna().sum() > 0:
                if self.nan_fill_method == 'median':
                    fill_value = df[col].median()
                elif self.nan_fill_method == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
        return df


    def load_multiple_tickers(self, tickers):
        """
        Загружает и объединяет данные для списка тикеров.
        
        Параметры:
            tickers - список тикеров (например, ["SBER", "GAZP", ...])
        
        Возвращает:
            overall_df - совокупный DataFrame, содержащий данные для всех тикеров с колонкой 'ticker'
        """
        overall_df = pd.DataFrame()
        for ticker in tickers:
            df = self.load_ticker_data(ticker)

            df = self._fill_nan_in_columns(df)
            overall_df = pd.concat([overall_df, df], ignore_index=True)

        overall_df['date'] = pd.to_datetime(overall_df['date'])
        overall_df.sort_values(['ticker', 'date'], inplace=True)

        return overall_df
    
