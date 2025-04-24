import pandas as pd
import numpy as np
import os
import logging
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
    current_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.logger import BaseLogger

class DataIntegrator(BaseLogger):
    def __init__(self, base_path, nan_fill_method='median'):
        """
        Параметры:
            base_path - базовый путь до папки с данными, в которой находятся подпапки:
                        AFKS/news_analysis и AFKS/tech_analysis
        """
        super().__init__('DataIntegrator')
        self.base_path = base_path
        self.nan_fill_method = nan_fill_method
        self.logger = self._setup_logger()

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

        files_exist = True
        if not os.path.exists(ml_path):
            self.logger.warning(f"Файл ML-признаков не найден: {ml_path}")
            files_exist = False
        if not os.path.exists(sentiment_path):
            self.logger.warning(f"Файл с настроениями не найден: {sentiment_path}")
            files_exist = False
        if not os.path.exists(tech_path):
            self.logger.warning(f"Файл с техническими индикаторами не найден: {tech_path}")
            files_exist = False

        if not files_exist:
            self.logger.info(f"Пропускаем тикер {ticker} из-за отсутствия необходимых файлов")
            return pd.DataFrame()

        try:
            ml_features = pd.read_csv(ml_path)
            sentiment_features = pd.read_csv(sentiment_path)
            base_with_tech = pd.read_csv(tech_path)
            
            ml_features['date'] = pd.to_datetime(ml_features['date']).dt.date
            sentiment_features['date'] = pd.to_datetime(sentiment_features['date']).dt.date
            base_with_tech['date'] = pd.to_datetime(base_with_tech['date']).dt.date
            
            df1 = pd.merge(base_with_tech, sentiment_features, on='date', how='left', suffixes=('', '_sentiment'))
            df = pd.merge(df1, ml_features, on='date', how='left', suffixes=('', '_ml'))
            
            cols_to_drop = [col for col in df.columns if col.endswith(('_sentiment', '_ml'))]
            
            base_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
            for col in base_cols:
                duplicates = [c for c in df.columns if c != col and c.startswith(col + '_')]
                cols_to_drop.extend(duplicates)
            
            df = df.drop(columns=cols_to_drop, errors='ignore')
            self.logger.info(f"Данные для {ticker} загружены за период: {df['date'].min()} - {df['date'].max()}")

            df['ticker'] = ticker

            return df
        
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных для {ticker}: {e}")
            return pd.DataFrame()
        
    def _fill_nan_in_columns(self, df):
        """
        Заполняет пропуски для указанных колонок (если заданы) или
        для всех числовых колонок, исключая базовые: date, open, close, volume, ticker.
        Метод заполнения определяется параметром nan_fill_method.
        """
        if df.empty:
            return df
            
        base_cols = ['date', 'open', 'close', 'volume', 'ticker']
        num_cols = df.select_dtypes(include=[np.number]).columns.difference(base_cols)
        
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            
            for col in num_cols:
                if col in df.columns and df.loc[ticker_mask, col].isna().any():
                    if self.nan_fill_method == 'median':
                        fill_value = df.loc[ticker_mask, col].median()
                    elif self.nan_fill_method == 'mean':
                        fill_value = df.loc[ticker_mask, col].mean()
                    elif self.nan_fill_method == 'zero':
                        fill_value = 0
                    else:
                        fill_value = 0
                    
                    if pd.isna(fill_value):
                        fill_value = 0
                        
                    df.loc[ticker_mask, col] = df.loc[ticker_mask, col].fillna(fill_value)
        
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
            self.logger.info(f"Обработка тикера {ticker}...")
            df = self.load_ticker_data(ticker)
            
            if not df.empty:
                null_count = df.isnull().sum().sum()
                if null_count > 0:
                    self.logger.info(f"  - Обнаружено {null_count} пропущенных значений для {ticker}")
                    df = self._fill_nan_in_columns(df)
                    self.logger.info(f"  - Пропущенные значения заполнены методом {self.nan_fill_method}")
                
                overall_df = pd.concat([overall_df, df], ignore_index=True)
            else:
                self.logger.info(f"  - Пропуск тикера {ticker}: нет данных")
        
        if not overall_df.empty:
            overall_df['date'] = pd.to_datetime(overall_df['date'])
            overall_df.sort_values(['ticker', 'date'], inplace=True)
            
            remaining_nulls = overall_df.isnull().sum().sum()
            if remaining_nulls > 0:
                self.logger.warning(f"Внимание: в итоговом датасете осталось {remaining_nulls} пропущенных значений")
                null_columns = overall_df.columns[overall_df.isnull().any()]
                self.logger.warning(f"Колонки с пропущенными значениями: {null_columns.tolist()}")
                overall_df = overall_df.fillna(0)
                self.logger.info("Все оставшиеся пропуски заполнены нулями")
        
        return overall_df
    
    def run_pipeline(self, tickers, output_path):
        """
        Запускает полный процесс загрузки, объединения данных и сохранения результата.

        Параметры:
            tickers (list): Список тикеров для обработки.
            output_path (str): Путь для сохранения объединенного CSV-файла.

        Возвращает:
            pd.DataFrame: Объединенный датасет или пустой DataFrame в случае ошибки.
        """
        combined_data = self.load_multiple_tickers(tickers=tickers)
        
        if not combined_data.empty:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            combined_data.to_csv(output_path, index=False)
            self.logger.info(f"Данные успешно объединены и сохранены в {output_path}")
            self.logger.info(f"Форма объединенного датасета: {combined_data.shape}")
            self.logger.info(f"Количество уникальных тикеров: {combined_data['ticker'].nunique()}")
        
def run_pipeline_integration(tickers, output_path="/Users/aeshef/Documents/GitHub/kursach/data/df.csv", method='zero'):
    DataIntegrator(base_path="/Users/aeshef/Documents/GitHub/kursach/data/processed_data", nan_fill_method=method).run_pipeline(
        tickers=tickers,
        output_path=output_path
    )
    
