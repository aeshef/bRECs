import os
import re
import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

class SmartLabYearlyParser:
    def __init__(self, ticker, base_path='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
        self.ticker = ticker.upper()
        self.url = f"https://smart-lab.ru/q/{self.ticker}/f/y/"
        self.base_path = base_path
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger('SmartLabYearlyParser')

    def fetch_page(self):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            ),
            "Accept-Language": "ru-RU,ru;q=0.9"
        }
        
        try:
            response = requests.get(self.url, headers=headers)
            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser")
            return None
        except Exception as e:
            self.logger.error(f"[{self.ticker}] Ошибка при запросе: {e}")
            return None

    def parse_yearly_tables(self, soup):
        if soup is None:
            return pd.DataFrame()

        tables = soup.find_all("table")
        if not tables:
            return pd.DataFrame()

        dataframes = []
        for table in tables:
            try:
                df_list = pd.read_html(str(table), header=None, decimal=',')
                if df_list:
                    dataframes.append(df_list[0])
            except Exception:
                continue

        if dataframes:
            return pd.concat(dataframes, ignore_index=True, sort=False)
        return pd.DataFrame()

    @staticmethod
    def to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(" ", "").replace("%", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return np.nan

    def clean_table(self, df):
        if df.empty:
            return df

        df_clean = df.copy()
        header_index = None
        
        for i in range(min(5, len(df_clean))):
            row = df_clean.iloc[i].astype(str).tolist()
            years_found = [cell for cell in row if re.search(r'\b\d{4}\b', cell)]
            if len(years_found) >= 2:
                header_index = i
                break

        if header_index is not None:
            new_header = df_clean.iloc[header_index].tolist()
            df_clean = df_clean.iloc[header_index+1:].reset_index(drop=True)
            df_clean.columns = new_header
        else:
            new_header = df_clean.iloc[0].tolist()
            df_clean = df_clean.iloc[1:].reset_index(drop=True)
            df_clean.columns = new_header

        seen = {}
        new_columns = []
        for item in df_clean.columns:
            if item in seen:
                seen[item] += 1
                new_columns.append(f"{item}_{seen[item]}")
            else:
                seen[item] = 0
                new_columns.append(str(item))
    
        df_clean.columns = new_columns
        return df_clean

    def preprocess_dataframe(self, df):
        if df.empty:
            return {}

        df_clean = self.clean_table(df)
        processed_data = {}
        
        indicator_col = df_clean.columns[0]
        needed_years = ['2023', '2024', '2025', '2026']

        for col in df_clean.columns[1:]:
            col_str = str(col).strip()
            match = re.search(r'\b(\d{4})\b', col_str)
            if match:
                year = match.group(1)
                if year in needed_years:  
                    year_df = df_clean[[indicator_col, col]].copy()
                    year_df.columns = ["Показатель", "Значение"]
                    year_df["Значение"] = year_df["Значение"].replace({"?": np.nan, "": np.nan})
                    year_df["value_float"] = year_df["Значение"].apply(self.to_float)
                    processed_data[year] = year_df
        
        return processed_data


def run_pipeline_fundamental(ticker_list, base_path='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
    """
    Основной процесс:
    1. Парсим данные для каждого тикера по годам
    2. Сохраняем два файла для каждого тикера и года:
       - {YEAR}/all.csv - все доступные показатели для этого тикера
       - {YEAR}/common.csv - общие показатели, которые есть у большинства тикеров
    3. Если у тикера нет каких-то "общих" показателей, заполняем их спецтокеном "NO_DATA"
    """
    print(f"\nОбработка тикеров: {ticker_list}")
    
    # Сбор данных по тикерам
    ticker_data_dict = {}
    parser_objects = {}

    for ticker in ticker_list:
        parser = SmartLabYearlyParser(ticker, base_path)
        parser_objects[ticker] = parser
        
        soup = parser.fetch_page()
        if soup is None:
            ticker_data_dict[ticker] = {}
            continue
        
        df = parser.parse_yearly_tables(soup)
        if df.empty:
            ticker_data_dict[ticker] = {}
            continue
        
        processed_data = parser.preprocess_dataframe(df)
        ticker_data_dict[ticker] = processed_data
    
    # Отладка: показатели для каждого тикера и года
    print("\n[DEBUG] Список показателей (value_float не NaN) для каждого тикера и года:")
    for ticker in ticker_list:
        data_for_ticker = ticker_data_dict.get(ticker, {})
        if not data_for_ticker:
            print(f"  {ticker} -> Данных нет совсем.")
            continue
        
        for year, df_year in data_for_ticker.items():
            inds = df_year[df_year["value_float"].notna()]["Показатель"].unique()
            print(f"  {ticker}, {year} -> {len(inds)} показателей: {list(inds)}")

    # Собираем все годы из всех тикеров
    all_years = set()
    for t in ticker_data_dict:
        all_years.update(ticker_data_dict[t].keys())

    # Для каждого года:
    # 1. Соберем все индикаторы для всех тикеров
    # 2. Найдем частоту встречаемости каждого индикатора
    # 3. Выберем индикаторы, которые встречаются у большинства тикеров
    
    # year -> {"ind1": count, "ind2": count, ...} где count - сколько тикеров имеют этот индикатор
    indicators_frequency_by_year = {}
    
    # year -> список "часто встречающихся" индикаторов
    common_indicators_by_year = {}
    
    for year in sorted(all_years):
        indicators_counter = Counter()
        
        # Собираем, сколько тикеров имеют каждый индикатор
        for ticker in ticker_list:
            data_for_year = ticker_data_dict[ticker].get(year, None)
            if data_for_year is not None:
                df_year = data_for_year
                available = set(df_year[df_year["value_float"].notna()]["Показатель"].unique())
                for indicator in available:
                    indicators_counter[indicator] += 1
        
        indicators_frequency_by_year[year] = indicators_counter
        
        # Считаем минимальное количество тикеров для "общности"
        # Например, если есть 16 тикеров, а данные есть только у 13, 
        # то берем индикаторы, которые есть хотя бы у 13 тикеров
        non_empty_tickers = sum(1 for ticker in ticker_list 
                               if year in ticker_data_dict.get(ticker, {}))
        
        # Если у всех тикеров есть этот год, то берем индикаторы, которые есть у всех
        # Иначе берем те, которые есть хотя бы у 90% тикеров с данными
        if non_empty_tickers == len(ticker_list):
            min_count = non_empty_tickers  # Все тикеры
        else:
            min_count = max(int(non_empty_tickers * 0.9), 1)  # Минимум 90% или 1
        
        # Отбираем индикаторы, которые встречаются хотя бы у min_count тикеров
        common_inds = [ind for ind, count in indicators_counter.items() 
                       if count >= min_count]
        
        common_indicators_by_year[year] = common_inds
    
    # Сохраняем данные в два файла для каждого тикера и года
    for ticker in ticker_list:
        parser = parser_objects[ticker]
        ticker_year_data = ticker_data_dict.get(ticker, {})

        for year in sorted(all_years):
            # Создаем папку для года
            year_folder_path = os.path.join(parser.base_path, ticker, "fundamental_analysis", year)
            os.makedirs(year_folder_path, exist_ok=True)
            
            # Пути к файлам
            all_file_path = os.path.join(year_folder_path, "all.csv")
            common_file_path = os.path.join(year_folder_path, "common.csv")
            
            # 1. Сохраняем "all.csv" - все показатели для данного тикера
            if year in ticker_year_data:
                df_year = ticker_year_data[year].copy()
                # Берем только не-NaN значения
                all_df = df_year[df_year["value_float"].notna()][["Показатель", "value_float"]]
                all_df.to_csv(all_file_path, index=False, encoding="utf-8")
            else:
                # Если данных нет, создаем пустой файл
                pd.DataFrame(columns=["Показатель", "value_float"]).to_csv(
                    all_file_path, index=False, encoding="utf-8"
                )
            
            # 2. Сохраняем "common.csv" - общие показатели для всех тикеров
            common_inds = common_indicators_by_year[year]
            
            if year in ticker_year_data:
                df_year = ticker_year_data[year].copy()
                
                # Создаем DataFrame с общими показателями
                result_rows = []
                
                for indicator in sorted(common_inds):
                    # Проверяем, есть ли этот индикатор у тикера
                    matching_rows = df_year[df_year["Показатель"] == indicator]
                    
                    if len(matching_rows) > 0 and not pd.isna(matching_rows["value_float"].iloc[0]):
                        # Индикатор есть и значение не NaN
                        result_rows.append({
                            "Показатель": indicator,
                            "value_float": matching_rows["value_float"].iloc[0]
                        })
                    else:
                        # Индикатор отсутствует или значение NaN
                        result_rows.append({
                            "Показатель": indicator,
                            "value_float": "NO_DATA"
                        })
                
                result_df = pd.DataFrame(result_rows)
                result_df.to_csv(common_file_path, index=False, encoding="utf-8")
                
            else:
                # Если данных нет, создаем файл со спецтокенами
                placeholder_df = pd.DataFrame({
                    "Показатель": common_inds,
                    "value_float": ["NO_DATA"] * len(common_inds)
                })
                placeholder_df.to_csv(common_file_path, index=False, encoding="utf-8")

    # Печатаем статистику
    print("\nРезультаты по годам:")
    for year in sorted(all_years):
        non_empty_tickers = sum(1 for ticker in ticker_list 
                               if year in ticker_data_dict.get(ticker, {}))
        
        common_inds = common_indicators_by_year[year]
        
        print(f"  Год {year}:")
        print(f"    - Данные есть у {non_empty_tickers} из {len(ticker_list)} тикеров")
        print(f"    - {len(common_inds)} общих показателей (для большинства тикеров):")
        
        for ind in sorted(common_inds):
            freq = indicators_frequency_by_year[year][ind]
            print(f"      * {ind} (встречается у {freq} тикеров)")

    print("\nПроцесс завершён. Созданы файлы all.csv и common.csv для каждого года и тикера.")
    print("Для отсутствующих показателей использовался токен 'NO_DATA'.")
