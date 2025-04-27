import tinkoff.invest
from tinkoff.invest import PortfolioRequest, PortfolioPosition, Client, RequestError, CandleInterval, HistoricCandle, \
    OrderType, OrderDirection, Quotation, InstrumentIdType, InstrumentStatus
from tinkoff.invest.services import InstrumentsService
from datetime import datetime
import pandas as pd
import os
import requests
import time
import zipfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
#     current_dir = os.path.dirname(current_dir)
#     if current_dir == os.path.dirname(current_dir):
#         break

# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# from utils.logger import BaseLogger

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class DataStorage(BaseLogger):
    """Класс для управления хранением данных"""
    
    def __init__(self, base_directory):
        """
        Инициализация хранилища данных
        
        :param base_directory: Корневая директория для хранения данных
        """
        super().__init__('DataStorage')
        self.base_directory = base_directory
        self.raw_data_dir = os.path.join(base_directory, "raw_data")
        self.processed_data_dir = os.path.join(base_directory, "processed_data")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def get_ticker_raw_path(self, ticker):
        """Получить путь к папке сырых данных для тикера"""
        path = os.path.join(self.raw_data_dir, ticker)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_ticker_processed_path(self, ticker):
        """Получить путь к папке обработанных данных для тикера"""
        path = os.path.join(self.processed_data_dir, ticker)
        os.makedirs(path, exist_ok=True)
        return path
    
    def store_raw_data(self, ticker, figi, year, content):
        """Сохранить сырые данные из API"""
        ticker_dir = self.get_ticker_raw_path(ticker)
        file_path = os.path.join(ticker_dir, f"{figi}_{year}.zip")
        
        with open(file_path, 'wb') as file:
            file.write(content)
        
        self.logger.debug(f"Saved raw data for {ticker}, {figi}, {year} to {file_path}")
        return file_path
    
    def extract_raw_data(self, ticker):
        """Распаковать все ZIP-файлы для тикера"""
        ticker_dir = self.get_ticker_raw_path(ticker)
        extracted_files = []
        
        for file in os.listdir(ticker_dir):
            if file.endswith(".zip"):
                zip_path = os.path.join(ticker_dir, file)
                extract_dir = os.path.join(ticker_dir, file.replace(".zip", ""))
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    for root, _, files in os.walk(extract_dir):
                        for csv_file in files:
                            if csv_file.endswith(".csv"):
                                extracted_files.append(os.path.join(root, csv_file))
                                
                except zipfile.BadZipFile:
                    self.logger.error(f"Bad zip file: {zip_path}")
        
        self.logger.info(f"Extracted {len(extracted_files)} CSV files for {ticker}")
        return extracted_files
    
    def load_raw_csv_files(self, csv_files):
        """Загрузить данные из CSV-файлов в единый DataFrame"""
        dfs = []
        
        for file_path in csv_files:
            try:
                temp_df = pd.read_csv(file_path, sep=";",
                                      names=["date", "open", "close", "min", "max", "volume", "unused"], 
                                      header=None)
                
                if 'unused' in temp_df.columns:
                    temp_df.drop('unused', axis=1, inplace=True)
                
                dfs.append(temp_df)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
        
        if not dfs:
            self.logger.warning("No valid CSV data files found")
            return None
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y-%m-%dT%H:%M:%SZ")
        combined_df.sort_values(by='date', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        self.logger.info(f"Loaded {len(combined_df)} rows from {len(dfs)} CSV files")
        return combined_df
    
    def save_processed_data(self, ticker, new_data):
        """
        Сохранить обработанные данные в один паркет-файл.
        Если файл уже существует, объединяем с новыми данными, удаляем дубликаты и перезаписываем
        :param ticker: Тикер инструмента
        :param new_data: DataFrame с новыми данными
        :return: Путь к сохранённому файлу
        """
        ticker_dir = self.get_ticker_processed_path(ticker)
        file_name = f"{ticker}.parquet"
        file_path = os.path.join(ticker_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                existing_data = pd.read_parquet(file_path)
                self.logger.debug(f"Found existing parquet file for {ticker} with {len(existing_data)} rows")
            except Exception as e:
                self.logger.error(f"Error reading existing parquet for {ticker}: {e}")
                existing_data = None
        else:
            existing_data = None
        
        if existing_data is not None and not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.drop_duplicates(subset=['date'], inplace=True)
            combined_data.sort_values(by='date', inplace=True)
            combined_data.reset_index(drop=True, inplace=True)
            self.logger.info(f"Combined {len(new_data)} new rows with {len(existing_data)} existing rows for {ticker}")
        else:
            combined_data = new_data.copy()
            self.logger.info(f"Created new dataset with {len(new_data)} rows for {ticker}")
        
        combined_data.to_parquet(file_path, index=False)
        self.logger.info(f"Saved processed data for {ticker} to {file_path}")
        return file_path

    
    def load_processed_data(self, ticker):
        """Загрузить обработанные данные для тикера из фиксированного файла"""
        ticker_dir = self.get_ticker_processed_path(ticker)
        file_name = f"{ticker}.parquet"
        file_path = os.path.join(ticker_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                data = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {len(data)} rows of processed data for {ticker}")
                return data
            except Exception as e:
                self.logger.error(f"Error reading processed data for {ticker}: {e}")
                return None
        self.logger.info(f"No processed data file found for {ticker}")
        return None

    
    def get_missing_date_ranges(self, ticker, start_date, end_date):
        """
        Определить, какие периоды данных отсутствуют для тикера
        
        :return: Список кортежей (start_date, end_date) для отсутствующих периодов
        """
        existing_data = self.load_processed_data(ticker)
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        if existing_data is None or len(existing_data) == 0:
            self.logger.info(f"No existing data for {ticker}, need to download complete range {start_date} to {end_date}")
            return [(start_date, end_date)]
        
        existing_dates = existing_data['date']
        min_date = existing_dates.min()
        max_date = existing_dates.max()
        
        missing_ranges = []
        if start_date < min_date:
            missing_ranges.append((start_date, min_date))
            self.logger.info(f"Missing date range for {ticker}: {start_date} to {min_date}")
        if end_date > max_date:
            missing_ranges.append((max_date, end_date))
            self.logger.info(f"Missing date range for {ticker}: {max_date} to {end_date}")
        
        if not missing_ranges:
            self.logger.info(f"All requested dates for {ticker} ({start_date} to {end_date}) are available")
            
        return missing_ranges


class TimeframeConverter:
    """Класс для преобразования временных интервалов"""
    
    @staticmethod
    def resample_ohlcv(df, timeframe):
        """
        Преобразовать минутные данные OHLCV в другой временной интервал
        
        :param df: DataFrame с минутными данными (колонки: date, open, close, min, max, volume)
        :param timeframe: Целевой временной интервал ('5min', '15min', '1h', '1d', '1w', '1M')
        :return: DataFrame с преобразованными данными
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        required_columns = ['date', 'open', 'close', 'min', 'max', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")

        df = df.copy()
        df.set_index('date', inplace=True)
        
        agg_dict = {
            'open': 'first',
            'min': 'min',
            'max': 'max',
            'close': 'last',
            'volume': 'sum'
        }
        
        resample_map = {
            '5min': '5T',
            '15min': '15T',
            '1h': 'H',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        }
        
        rule = resample_map.get(timeframe, '1D')
        resampled = df.resample(rule).agg(agg_dict)
        resampled.reset_index(inplace=True)
        
        return resampled
    

class TinkoffDataDownloader(BaseLogger):
    """Класс для скачивания данных из Tinkoff API"""
    
    def __init__(self, token, storage):
        """
        Инициализация загрузчика данных
        
        :param token: API токен Tinkoff
        :param storage: Экземпляр класса DataStorage
        """
        super().__init__('TinkoffDataDownloader')
        self.url = "https://invest-public-api.tinkoff.ru/history-data"
        self.token = token
        self.storage = storage
        self.df_screener_all = None
        self.df_screener_rub = None
        self.initialize_instruments()
        
    def initialize_instruments(self):
        """Получение списка доступных инструментов"""
        self.logger.info("Initializing instruments from Tinkoff API")
        try:
            with Client(token=self.token) as client:
                instruments = client.instruments
                self.df_screener_all = pd.DataFrame(
                    instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL).instruments,
                    columns=["name", "ticker", "uid", "figi", "isin", "lot", "currency"])
                self.df_screener_rub = self.df_screener_all[self.df_screener_all["currency"] == "rub"]
                self.logger.info(f"Successfully loaded {len(self.df_screener_all)} instruments, {len(self.df_screener_rub)} in RUB")
        except Exception as e:
            self.logger.error(f"Failed to initialize instruments: {e}")
    
    def get_figi(self, ticker) -> list:
        """Получение FIGI для тикера"""
        if not self.df_screener_rub[self.df_screener_rub["ticker"] == ticker].empty:
            ticker_figi = self.df_screener_rub[self.df_screener_rub["ticker"] == ticker]
            
            if ticker_figi.shape[0] > 1:
                figi_list = list(ticker_figi["figi"])
                self.logger.info(f"Found multiple FIGI for {ticker}: {figi_list}")
                return figi_list
            else:
                figi = ticker_figi["figi"].iloc[0]
                self.logger.info(f"Found FIGI for {ticker}: {figi}")
                return [figi]
        self.logger.warning(f"No FIGI found for ticker {ticker}")
        return []
        
    def get_uid(self, ticker) -> list:
        """Получение UID для тикера"""
        if not self.df_screener_rub[self.df_screener_rub["ticker"] == ticker].empty:
            ticker_uid = self.df_screener_rub[self.df_screener_rub["ticker"] == ticker]
            
            if ticker_uid.shape[0] > 1:
                uid_list = list(ticker_uid["uid"])
                self.logger.info(f"Found multiple UIDs for {ticker}: {uid_list}")
                return uid_list
            else:
                uid = ticker_uid["uid"].iloc[0]
                self.logger.info(f"Found UID for {ticker}: {uid}")
                return [uid]
        self.logger.warning(f"No UID found for ticker {ticker}")
        return []
        
    def get_isin(self, ticker) -> list:
        """Получение ISIN для тикера"""
        if not self.df_screener_rub[self.df_screener_rub["ticker"] == ticker].empty:
            ticker_isin = self.df_screener_rub[self.df_screener_rub["ticker"] == ticker]
            
            if ticker_isin.shape[0] > 1:
                isin_list = list(ticker_isin["isin"])
                self.logger.info(f"Found multiple ISINs for {ticker}: {isin_list}")
                return isin_list
            else:
                isin = ticker_isin["isin"].iloc[0]
                self.logger.info(f"Found ISIN for {ticker}: {isin}")
                return [isin]
        self.logger.warning(f"No ISIN found for ticker {ticker}")
        return []

    def is_figi_correct(self, figi) -> bool:
        """Проверка корректности FIGI"""
        correct_figi = False
        figi_path = os.path.join(self.storage.base_directory, "figi.txt")
        if os.path.exists(figi_path):
            with open(figi_path, 'r') as file:
                for line in file:
                    if figi in line:
                        correct_figi = True
                        self.logger.debug(f"FIGI {figi} validated as correct from figi.txt")
        return correct_figi
        
    def get_correct_figi(self, figi_list) -> str:
        """Получение корректного FIGI из списка"""
        for figi in figi_list:
            if self.is_figi_correct(figi):
                self.logger.info(f"Found correct FIGI: {figi}")
                return figi
        self.logger.warning("No correct FIGI found in the list")
        return "0"
    
    def download_for_ticker(self, ticker, start_date, end_date):
        """
        Скачать данные для тикера за указанный период
        
        :param ticker: Тикер инструмента
        :param start_date: Дата начала в формате YYYY-MM-DD
        :param end_date: Дата окончания в формате YYYY-MM-DD
        :return: DataFrame с минутными данными
        """
        self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        years_to_download = list(range(start_year, end_year + 1))
        
        ticker_figi = self.get_figi(ticker)
        ticker_uid = self.get_uid(ticker)
        ticker_isin = self.get_isin(ticker)
        
        correct_figi = self.get_correct_figi(ticker_figi)
        
        downloaded_files = []
        
        if correct_figi != "0":
            self.logger.info(f"Using correct FIGI {correct_figi} for {ticker}")
            
            for year in years_to_download:
                file_path = self._download_year_data(ticker, correct_figi, year)
                if file_path:
                    downloaded_files.append(file_path)
        else:
            self.logger.info(f"No correct FIGI found, trying all available identifiers for {ticker}")
            for instrument_list, id_type in zip([ticker_figi, ticker_isin, ticker_uid], ["FIGI", "ISIN", "UID"]):
                for instrument_id in instrument_list:
                    self.logger.info(f"Trying {id_type} {instrument_id} for {ticker}")
                    for year in years_to_download:
                        file_path = self._download_year_data(ticker, instrument_id, year)
                        if file_path:
                            downloaded_files.append(file_path)
        
        self.logger.info(f"Downloaded {len(downloaded_files)} files for {ticker}")
        
        csv_files = self.storage.extract_raw_data(ticker)
        data = self.storage.load_raw_csv_files(csv_files)
        
        if data is not None:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            
            data = data[(data['date'] >= start_date_dt) & (data['date'] <= end_date_dt)]
            self.logger.info(f"Filtered data for {ticker} within range, got {len(data)} rows")

            self.storage.save_processed_data(ticker, data)
            self._log_data_stats(ticker, data)
        else:
            self.logger.warning(f"No data loaded for {ticker}")
        
        return data
    
    def _download_year_data(self, ticker, figi, year):
        """Скачать данные за определенный год"""
        self.logger.info(f"Downloading {ticker} ({figi}) for year {year}")
        
        params = {"figi": figi, "year": year, "interval": "1min"}
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(self.url, params=params, headers=headers)
            
            if response.status_code == 200:
                self.logger.info(f"Successfully downloaded data for {ticker} ({figi}) in {year}")
                return self.storage.store_raw_data(ticker, figi, year, response.content)
            elif response.status_code == 429:
                self.logger.warning("Rate limit exceeded. Sleeping for 5 seconds...")
                time.sleep(5)
                return self._download_year_data(ticker, figi, year)
            elif response.status_code == 404:
                self.logger.warning(f"No data found for {ticker} ({figi}) in {year}")
                return None
            else:
                self.logger.error(f"Error downloading data: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception during download: {e}")
            return None
    
    def _log_data_stats(self, ticker, data):
        """Вывести статистику о скачанных данных"""
        if data is None or len(data) == 0:
            self.logger.warning(f"{ticker}: No data found")
            return
            
        self.logger.info(f"{ticker} data statistics:")
        self.logger.info(f"  - Total rows: {len(data)}")
        self.logger.info(f"  - First date: {data['date'].min()}")
        self.logger.info(f"  - Last date: {data['date'].max()}")

class MarketDataManager(BaseLogger):
    """Основной класс для управления рыночными данными"""
    
    def __init__(self, token, base_directory):
        """
        Инициализация менеджера данных
        
        :param token: API токен Tinkoff
        :param base_directory: Корневая директория для хранения данных
        """
        super().__init__('MarketDataManager')
        self.storage = DataStorage(base_directory)
        self.downloader = TinkoffDataDownloader(token, self.storage)
        self.converter = TimeframeConverter()
        self.logger.info("MarketDataManager initialized")
    
    def get_data(self, ticker, start_date, end_date, timeframe="1min", force_download=False):
        """
        Получить данные для тикера с указанным временным интервалом
        
        :param ticker: Тикер инструмента
        :param start_date: Дата начала в формате YYYY-MM-DD
        :param end_date: Дата окончания в формате YYYY-MM-DD
        :param timeframe: Временной интервал ('1min', '5min', '15min', '1h', '1d', '1w', '1M')
        :param force_download: Принудительно скачать новые данные, даже если они уже есть
        :return: DataFrame с данными
        """
        self.logger.info(f"Getting data for {ticker} from {start_date} to {end_date} with timeframe {timeframe}")
        if force_download:
            self.logger.info(f"Force download enabled for {ticker}")
            existing_data = None
        else:
            existing_data = self.storage.load_processed_data(ticker)
        
        if existing_data is not None:
            self.logger.info(f"Found existing data for {ticker}, checking for missing ranges")
            missing_ranges = self.storage.get_missing_date_ranges(ticker, start_date, end_date)
            
            for range_start, range_end in missing_ranges:
                self.logger.info(f"Downloading missing data for {ticker} from {range_start} to {range_end}")
                new_data = self.downloader.download_for_ticker(ticker, range_start.strftime('%Y-%m-%d'), 
                                                             range_end.strftime('%Y-%m-%d'))
                
                if new_data is not None and len(new_data) > 0:
                    self.logger.info(f"Merging {len(new_data)} new rows with existing data for {ticker}")
                    existing_data = pd.concat([existing_data, new_data], ignore_index=True)
                    existing_data.drop_duplicates(subset=['date'], inplace=True)
                    existing_data.sort_values(by='date', inplace=True)
                    existing_data.reset_index(drop=True, inplace=True)
            
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            data = existing_data[(existing_data['date'] >= start_date_dt) & 
                                 (existing_data['date'] <= end_date_dt)]
            self.logger.info(f"Filtered existing data to requested date range, got {len(data)} rows")
        else:
            self.logger.info(f"No existing data found for {ticker}, downloading from scratch")
            data = self.downloader.download_for_ticker(ticker, start_date, end_date)
        
        if timeframe != "1min" and data is not None and len(data) > 0:
            self.logger.info(f"Resampling data from 1min to {timeframe} for {ticker}")
            data = self.converter.resample_ohlcv(data, timeframe)
            self.logger.info(f"Resampled to {len(data)} rows at {timeframe} timeframe")
        
        return data
    
    def get_data_for_multiple_tickers(self, tickers, start_date, end_date, timeframe="1min", max_workers=5):
        """
        Получить данные для нескольких тикеров с использованием многопоточности
        
        :param tickers: Список тикеров
        :param start_date: Дата начала в формате YYYY-MM-DD
        :param end_date: Дата окончания в формате YYYY-MM-DD
        :param timeframe: Временной интервал ('1min', '5min', '15min', '1h', '1d', '1w', '1M')
        :param max_workers: Максимальное количество потоков
        :return: Словарь {ticker: DataFrame}
        """
        self.logger.info(f"Getting data for {len(tickers)} tickers with {max_workers} workers")
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.get_data, ticker, start_date, end_date, timeframe): ticker 
                for ticker in tickers
            }
            
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results[ticker] = data
                    
                    if data is not None and len(data) > 0:
                        self.logger.info(f"{ticker} data successfully retrieved:")
                        self.logger.info(f"  - Timeframe: {timeframe}")
                        self.logger.info(f"  - Total rows: {len(data)}")
                        self.logger.info(f"  - First date: {data['date'].min()}")
                        self.logger.info(f"  - Last date: {data['date'].max()}")
                    else:
                        self.logger.warning(f"{ticker}: No data available")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {ticker}: {e}")
                    results[ticker] = None
        
        self.logger.info(f"Completed data retrieval for {len(tickers)} tickers")
        return results

class MarketDataPipeline(BaseLogger):
    """Класс для запуска пайплайна загрузки рыночных данных"""
    
    def __init__(self, base_directory=f"/{BASE_PATH}/data"):
        """Инициализация пайплайна с директорией для данных"""
        super().__init__('MarketDataPipeline')
        self.base_directory = base_directory
        self.logger.info(f"MarketDataPipeline initialized with base directory: {base_directory}")
    
    def run(self, tickers, start_date, end_date, token, timeframe="1min"):
        """s
        Запускает пайплайн: настраивает логирование и загружает данные.

        Параметры:
            tickers (list): Список тикеров.
            start_date (str): Начальная дата в формате "YYYY-MM-DD".
            end_date (str): Конечная дата в формате "YYYY-MM-DD".
            token (str): API токен Tinkoff.
            timeframe (str): Временной интервал, по умолчанию "1min".
            
        Возвращает:
            Данные, полученные через get_data_for_multiple_tickers.
        """
        self.logger.info(f"Starting market data pipeline for {len(tickers)} tickers")
        self.logger.info(f"Date range: {start_date} to {end_date}, timeframe: {timeframe}")
        
        manager = MarketDataManager(
            token=token,
            base_directory=self.base_directory
        )   

        self.log_start_process("Data download")
        data = manager.get_data_for_multiple_tickers(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        self.log_end_process("Data download")

        success_count = sum(1 for df in data.values() if df is not None and len(df) > 0)
        self.logger.info(f"Successfully downloaded data for {success_count} out of {len(tickers)} tickers")
        
        return data


def run_pipeline_market(tickers, start_date, end_date, token, timeframe="1min"):
    pipeline = MarketDataPipeline()
    return pipeline.run(tickers, start_date, end_date, token, timeframe)
