import pandas as pd
import numpy as np
import os

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

        # Проверяем существование файлов
        files_exist = True
        if not os.path.exists(ml_path):
            print(f"Файл ML-признаков не найден: {ml_path}")
            files_exist = False
        if not os.path.exists(sentiment_path):
            print(f"Файл с настроениями не найден: {sentiment_path}")
            files_exist = False
        if not os.path.exists(tech_path):
            print(f"Файл с техническими индикаторами не найден: {tech_path}")
            files_exist = False
        
        if not files_exist:
            print(f"Пропускаем тикер {ticker} из-за отсутствия файлов")
            return pd.DataFrame()

        # Загружаем данные
        try:
            ml_features = pd.read_csv(ml_path)
            sentiment_features = pd.read_csv(sentiment_path)
            base_with_tech = pd.read_csv(tech_path)
            
            # Убедимся, что формат даты одинаковый
            ml_features['date'] = pd.to_datetime(ml_features['date']).dt.date
            sentiment_features['date'] = pd.to_datetime(sentiment_features['date']).dt.date
            base_with_tech['date'] = pd.to_datetime(base_with_tech['date']).dt.date
            
            # Последовательное объединение
            df1 = pd.merge(base_with_tech, sentiment_features, on='date', how='left', suffixes=('', '_sentiment'))
            df = pd.merge(df1, ml_features, on='date', how='left', suffixes=('', '_ml'))
            
            # Очистка дублирующихся колонок
            # Создаем список колонок для удаления - все с суффиксами
            cols_to_drop = [col for col in df.columns if col.endswith(('_sentiment', '_ml'))]
            
            # Также проверим и удалим дублирующиеся колонки без явных суффиксов
            base_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
            for col in base_cols:
                duplicates = [c for c in df.columns if c != col and c.startswith(col + '_')]
                cols_to_drop.extend(duplicates)
            
            # Удаляем дубликаты
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            # Добавляем тикер
            df['ticker'] = ticker
            
            return df
        
        except Exception as e:
            print(f"Ошибка при загрузке данных для {ticker}: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Группировка по тикеру для правильного заполнения
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
                    
                    # Если после всего fill_value все еще NaN (например, если все значения NaN)
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
            print(f"Обработка тикера {ticker}...")
            df = self.load_ticker_data(ticker)
            
            if not df.empty:
                # Проверяем наличие пропущенных значений
                null_count = df.isnull().sum().sum()
                if null_count > 0:
                    print(f"  - Обнаружено {null_count} пропущенных значений для {ticker}")
                    df = self._fill_nan_in_columns(df)
                    print(f"  - Пропущенные значения заполнены методом {self.nan_fill_method}")
                
                overall_df = pd.concat([overall_df, df], ignore_index=True)
            else:
                print(f"  - Пропуск тикера {ticker}: нет данных")
        
        if not overall_df.empty:
            # Преобразуем дату в формат datetime для сортировки
            overall_df['date'] = pd.to_datetime(overall_df['date'])
            overall_df.sort_values(['ticker', 'date'], inplace=True)
            
            # Проверка на оставшиеся NaN
            remaining_nulls = overall_df.isnull().sum().sum()
            if remaining_nulls > 0:
                print(f"Внимание: в итоговом датасете осталось {remaining_nulls} пропущенных значений")
                
                # Вывод колонок с пропущенными значениями
                null_columns = overall_df.columns[overall_df.isnull().any()]
                print(f"Колонки с пропущенными значениями: {null_columns.tolist()}")
                
                # Заполняем оставшиеся пропуски нулями
                overall_df = overall_df.fillna(0)
                print("Все оставшиеся пропуски заполнены нулями")
        
        return overall_df
        
        
        
