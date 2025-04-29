import tinkoff.invest
from tinkoff.invest import Client, InstrumentStatus
from datetime import datetime
import pandas as pd
import os
import requests
import time
import zipfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
from typing import List, Tuple, Optional, Dict


from pys.data_collection.private_info import BASE_PATH
from pys.utils.logger import BaseLogger

# # ==============================================
# #      DataStorage Class (Unchanged)
# # ==============================================
# class DataStorage(BaseLogger):
#     def __init__(self, base_directory):
#         super().__init__('DataStorage')
#         self.base_directory = base_directory
#         self.raw_data_dir = os.path.join(base_directory, "raw_data")
#         self.processed_data_dir = os.path.join(base_directory, "processed_data")

#         os.makedirs(self.raw_data_dir, exist_ok=True)
#         os.makedirs(self.processed_data_dir, exist_ok=True)
#         self.logger.info(f"DataStorage initialized. Raw: {self.raw_data_dir}, Processed: {self.processed_data_dir}")

#     def get_ticker_raw_path(self, ticker):
#         path = os.path.join(self.raw_data_dir, ticker)
#         os.makedirs(path, exist_ok=True)
#         return path

#     def get_ticker_processed_path(self, ticker):
#         path = os.path.join(self.processed_data_dir, ticker)
#         os.makedirs(path, exist_ok=True)
#         return path

#     def get_raw_zip_filepath(self, ticker, figi, year):
#         """Получить ожидаемый путь к файлу ZIP сырых данных."""
#         ticker_dir = self.get_ticker_raw_path(ticker)
#         return os.path.join(ticker_dir, f"{figi}_{year}.zip")

#     def check_raw_zip_exists(self, ticker, figi, year):
#         """Проверить, существует ли файл ZIP сырых данных для года."""
#         filepath = self.get_raw_zip_filepath(ticker, figi, year)
#         exists = os.path.exists(filepath)
#         self.logger.debug(f"Checking for raw ZIP: {filepath} - Exists: {exists}")
#         return exists

#     def store_raw_data(self, ticker, figi, year, content):
#         """Сохранить сырые данные (ZIP) из API."""
#         file_path = self.get_raw_zip_filepath(ticker, figi, year)
#         try:
#             with open(file_path, 'wb') as file:
#                 file.write(content)
#             self.logger.debug(f"Saved raw data for {ticker}, {figi}, {year} to {file_path}")
#             return file_path
#         except IOError as e:
#             self.logger.error(f"Failed to save raw data to {file_path}: {e}")
#             return None

#     def extract_specific_raw_zips(self, zip_files_to_extract: List[str]) -> List[str]:
#         """
#         Распаковать только указанные ZIP-файлы.
#         Возвращает список путей к распакованным CSV файлам.
#         """
#         extracted_csv_files = []
#         if not zip_files_to_extract:
#             self.logger.info("No new ZIP files to extract.")
#             return []

#         self.logger.info(f"Extracting {len(zip_files_to_extract)} specific ZIP files...")
#         for zip_path in zip_files_to_extract:
#             if not os.path.exists(zip_path):
#                 self.logger.warning(f"ZIP file not found for extraction: {zip_path}")
#                 continue

#             extract_dir = zip_path.replace(".zip", "")
#             os.makedirs(extract_dir, exist_ok=True) # Ensure extract dir exists

#             try:
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     zip_ref.extractall(extract_dir)
#                 self.logger.debug(f"Extracted {zip_path} to {extract_dir}")

#                 for root, _, files in os.walk(extract_dir):
#                     for csv_file in files:
#                         if csv_file.endswith(".csv"):
#                             csv_path = os.path.join(root, csv_file)
#                             extracted_csv_files.append(csv_path)
#                             self.logger.debug(f"Found extracted CSV: {csv_path}")

#             except zipfile.BadZipFile:
#                 self.logger.error(f"Bad zip file, cannot extract: {zip_path}")
#             except Exception as e:
#                  self.logger.error(f"Error extracting {zip_path}: {e}")

#         self.logger.info(f"Extraction complete. Found {len(extracted_csv_files)} new CSV files.")
#         return extracted_csv_files

#     def load_specific_raw_csv_files(self, csv_files: List[str]) -> Optional[pd.DataFrame]:
#         """Загрузить данные только из указанных CSV-файлов в единый DataFrame."""
#         if not csv_files:
#             self.logger.info("No specific CSV files provided for loading.")
#             return None

#         dfs = []
#         self.logger.info(f"Loading data from {len(csv_files)} specified CSV files...")
#         for file_path in csv_files:
#             try:
#                 temp_df = pd.read_csv(file_path, sep=";",
#                                       names=["date", "open", "close", "min", "max", "volume", "unused"],
#                                       header=None, dtype={'volume': 'Int64'})

#                 if 'unused' in temp_df.columns:
#                     temp_df.drop('unused', axis=1, inplace=True)

#                  # Basic data validation - check if first column looks like a date string
#                 if isinstance(temp_df['date'].iloc[0], str) and not temp_df['date'].iloc[0].startswith('20'):
#                     self.logger.warning(f"Suspicious date format in {file_path}, skipping.")
#                     continue

#                 dfs.append(temp_df)
#                 self.logger.debug(f"Successfully loaded {len(temp_df)} rows from {file_path}")

#             except pd.errors.EmptyDataError:
#                  self.logger.warning(f"Empty CSV file encountered: {file_path}")
#             except IndexError: # Handles case where file is completely empty or malformed -> iloc[0] fails
#                  self.logger.warning(f"Could not read first row (IndexError), likely empty or malformed CSV: {file_path}")
#             except Exception as e:
#                 self.logger.error(f"Error reading {file_path}: {e}")

#         if not dfs:
#             self.logger.warning("No valid data loaded from the specified CSV files.")
#             return None

#         combined_df = pd.concat(dfs, ignore_index=True)
#         self.logger.debug(f"Combined {len(dfs)} dataframes, total rows before parsing: {len(combined_df)}")

#         combined_df["date"] = pd.to_datetime(combined_df["date"], format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
#         combined_df.dropna(subset=['date'], inplace=True)

#         for col in ["open", "close", "min", "max", "volume"]:
#             # Use float64 for price columns to handle potential NaNs before conversion if needed
#             combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
#             if col != 'volume':
#                  combined_df[col] = combined_df[col].astype('float64') # Ensure prices are float
#             else:
#                  combined_df[col] = combined_df[col].astype('Int64') # Keep volume as nullable Int

#         combined_df.sort_values(by='date', inplace=True)
#         combined_df.reset_index(drop=True, inplace=True)

#         self.logger.info(f"Loaded and processed {len(combined_df)} rows from {len(dfs)} CSV files.")
#         return combined_df

#     def save_processed_data(self, ticker: str, new_data: pd.DataFrame) -> str:
#         """
#         Сохранить обработанные данные. Объединяет с существующими данными, если они есть.
#         """
#         if new_data is None or new_data.empty:
#             self.logger.info(f"No new data provided for {ticker}, nothing to save/merge.")
#             return self.get_processed_filepath(ticker)

#         ticker_dir = self.get_ticker_processed_path(ticker)
#         file_path = self.get_processed_filepath(ticker)

#         existing_data = None
#         if os.path.exists(file_path):
#             try:
#                 existing_data = pd.read_parquet(file_path)
#                 self.logger.debug(f"Read existing parquet for {ticker}: {len(existing_data)} rows.")
#             except Exception as e:
#                 self.logger.error(f"Error reading existing parquet {file_path}: {e}. Will overwrite.")
#                 existing_data = None

#         if existing_data is not None and not existing_data.empty:
#             # Ensure correct dtypes before merge attempts, especially for comparison
#             if 'date' in existing_data.columns and not pd.api.types.is_datetime64_any_dtype(existing_data['date']):
#                 existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
#             if 'date' in new_data.columns and not pd.api.types.is_datetime64_any_dtype(new_data['date']):
#                  new_data['date'] = pd.to_datetime(new_data['date'], errors='coerce')

#             combined_data = pd.concat([existing_data, new_data], ignore_index=True)
#             self.logger.debug(f"Combined {len(new_data)} new rows with {len(existing_data)} existing rows for {ticker}. Total before dedup: {len(combined_data)}")

#             # Drop duplicates based on 'date'
#             combined_data.drop_duplicates(subset=['date'], keep='last', inplace=True) # Use 'last' to ensure newest data prevails
#             combined_data.sort_values(by='date', inplace=True)
#             combined_data.reset_index(drop=True, inplace=True)
#             self.logger.info(f"Merged and deduplicated data for {ticker}. Final rows: {len(combined_data)}")
#         else:
#             combined_data = new_data.copy()
#             # Ensure date is datetime type
#             if 'date' in combined_data.columns and not pd.api.types.is_datetime64_any_dtype(combined_data['date']):
#                  combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')
#             combined_data.sort_values(by='date', inplace=True)
#             combined_data.reset_index(drop=True, inplace=True)
#             self.logger.info(f"Saving new dataset for {ticker} with {len(combined_data)} rows.")

#         try:
#             combined_data.to_parquet(file_path, index=False, engine='pyarrow', compression='snappy')
#             self.logger.info(f"Successfully saved processed data for {ticker} to {file_path}")
#         except Exception as e:
#              self.logger.error(f"Error writing parquet file {file_path}: {e}")

#         return file_path

#     def get_processed_filepath(self, ticker):
#          """Получить путь к файлу обработанных данных (Parquet)."""
#          ticker_dir = self.get_ticker_processed_path(ticker)
#          return os.path.join(ticker_dir, f"{ticker}.parquet")

#     def load_processed_data(self, ticker: str) -> Optional[pd.DataFrame]:
#         """Загрузить обработанные данные для тикера из Parquet файла."""
#         file_path = self.get_processed_filepath(ticker)

#         if os.path.exists(file_path):
#             try:
#                 data = pd.read_parquet(file_path)
#                 if data.empty:
#                     self.logger.info(f"Loaded empty processed data file for {ticker}: {file_path}")
#                 else:
#                     # Ensure date is datetime
#                     if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
#                          data['date'] = pd.to_datetime(data['date'], errors='coerce')
#                          data.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

#                     self.logger.info(f"Loaded {len(data)} rows of processed data for {ticker} from {file_path}")
#                 return data
#             except Exception as e:
#                 self.logger.error(f"Error reading processed parquet {file_path}: {e}", exc_info=True) # Log traceback
#                 return None
#         else:
#             self.logger.info(f"No processed data file found for {ticker} at {file_path}")
#             return None

#     def get_missing_date_ranges(self, ticker: str, start_date_req: pd.Timestamp, end_date_req: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
#         """
#         Определить, какие периоды данных отсутствуют в обработанном файле
#         относительно запрошенного диапазона.
#         """
#         existing_data = self.load_processed_data(ticker)

#         # Ensure request dates are timezone-naive or consistent with data (assuming naive UTC here)
#         if start_date_req.tz is not None:
#             start_date_req = start_date_req.tz_localize(None)
#         if end_date_req.tz is not None:
#             end_date_req = end_date_req.tz_localize(None)


#         if existing_data is None or existing_data.empty:
#             self.logger.info(f"No existing data for {ticker}. Need full range: {start_date_req} to {end_date_req}")
#             return [(start_date_req, end_date_req)]

#         if not pd.api.types.is_datetime64_any_dtype(existing_data['date']):
#              self.logger.warning(f"Date column in existing data for {ticker} is not datetime type. Attempting conversion.")
#              existing_data['date'] = pd.to_datetime(existing_data['date'], errors='coerce')
#              existing_data.dropna(subset=['date'], inplace=True)
#              if existing_data.empty:
#                   self.logger.error(f"Date conversion failed or resulted in empty dataframe for {ticker}. Treating as no existing data.")
#                   return [(start_date_req, end_date_req)]

#         # Ensure existing dates are timezone-naive for comparison
#         if existing_data['date'].dt.tz is not None:
#              existing_data['date'] = existing_data['date'].dt.tz_localize(None)


#         min_date_local = existing_data['date'].min()
#         max_date_local = existing_data['date'].max()
#         self.logger.info(f"Existing data for {ticker} spans: {min_date_local} to {max_date_local}")

#         missing_ranges = []

#         # 1. Gap before existing data
#         if start_date_req < min_date_local:
#             # Request from start_date_req up to the day before min_date_local
#             # Ensure end of gap is not before start_date_req
#             end_of_gap = min(end_date_req, min_date_local - pd.Timedelta(microseconds=1)) # Just before min_date_local
#             if start_date_req <= end_of_gap:
#                  missing_ranges.append((start_date_req, end_of_gap))
#                  self.logger.info(f"Missing range detected for {ticker} (before): {start_date_req} to {end_of_gap}")

#         # 2. Gap after existing data
#         if end_date_req > max_date_local:
#              # Request from the moment after max_date_local up to end_date_req
#              # Ensure start of gap is not after end_date_req
#             start_of_gap = max(start_date_req, max_date_local + pd.Timedelta(microseconds=1)) # Just after max_date_local
#             if start_of_gap <= end_date_req:
#                 missing_ranges.append((start_of_gap, end_date_req))
#                 self.logger.info(f"Missing range detected for {ticker} (after): {start_of_gap} to {end_date_req}")

#         # Internal gaps require more complex logic (checking for time differences > expected interval)
#         # Skipping internal gap check for now as it's less common for yearly downloads

#         if not missing_ranges:
#             self.logger.info(f"No missing date ranges detected for {ticker} within requested period {start_date_req} to {end_date_req}, based on local min/max.")

#         return missing_ranges


# # ==============================================
# #      TimeframeConverter Class (Unchanged)
# # ==============================================
# class TimeframeConverter:
#     @staticmethod
#     def resample_ohlcv(df: Optional[pd.DataFrame], timeframe: str) -> pd.DataFrame:
#         """Преобразовать минутные данные OHLCV в другой временной интервал."""
#         if df is None or df.empty:
#             return pd.DataFrame()

#         required_columns = ['date', 'open', 'close', 'min', 'max', 'volume']
#         if not all(col in df.columns for col in required_columns):
#             # Log columns found for debugging
#             missing = set(required_columns) - set(df.columns)
#             raise ValueError(f"DataFrame для ресемплинга должен содержать колонки: {required_columns}. Отсутствуют: {missing}")

#         df = df.copy()
#         if not pd.api.types.is_datetime64_any_dtype(df['date']):
#              df['date'] = pd.to_datetime(df['date'])
#         df.set_index('date', inplace=True)

#         # Check for NaN values in essential columns before resampling
#         essential_cols = ['open', 'close', 'min', 'max']
#         nan_check = df[essential_cols].isnull().any()
#         if nan_check.any():
#             # Consider logging or deciding how to handle NaNs (e.g., ffill, skip rows)
#             # print(f"Warning: NaN values found in essential columns before resampling: \n{nan_check}")
#             # Option: df.dropna(subset=essential_cols, inplace=True) # Or use ffill/bfill
#             pass # Keep NaNs for now, resampler might handle them

#         agg_dict = {
#             'open': 'first',
#             'min': 'min',
#             'max': 'max',
#             'close': 'last',
#             'volume': 'sum'
#         }

#         resample_map = {'5min': '5T', '15min': '15T', '1h': 'H', '1d': 'D', '1w': 'W', '1M': 'MS'} # Use MS for MonthStart
#         rule = resample_map.get(timeframe)
#         if rule is None:
#              raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(resample_map.keys())}")

#         # Try resampling
#         try:
#             # origin='start' / 'epoch' might be useful depending on desired alignment
#             # Use label='left', closed='left' typical for financial OHLC
#             resampled = df.resample(rule, label='left', closed='left').agg(agg_dict)
#         except Exception as e:
#             print(f"Error during resampling: {e}")
#             # Provide more context if possible
#             # print("Input df info before error:")
#             # df.info()
#             # print(df.head())
#             raise # Re-raise the exception after printing info


#         resampled.dropna(subset=['open', 'close', 'min', 'max'], how='all', inplace=True)
#         resampled['volume'].fillna(0, inplace=True)
#         resampled['volume'] = resampled['volume'].astype('Int64') # Convert back to nullable Int

#         resampled.reset_index(inplace=True)

#         return resampled


# # ==============================================
# #      TinkoffDataDownloader Class (FIXED)
# # ==============================================
# class TinkoffDataDownloader(BaseLogger):
#     def __init__(self, token: str, storage: DataStorage):
#         super().__init__('TinkoffDataDownloader')
#         self.url = "https://invest-public-api.tinkoff.ru/history-data"
#         self.token = token
#         self.storage = storage
#         self.df_screener_all = pd.DataFrame() # Initialize as empty
#         self.df_screener_rub = pd.DataFrame() # Initialize as empty
#         self._initialize_instruments() # Call initialization

#     def _initialize_instruments(self):
#         """Получение списка доступных инструментов (приватный метод)."""
#         self.logger.info("Initializing instruments from Tinkoff API...")
#         try:
#             with Client(token=self.token) as client:
#                 instruments_service = client.instruments
#                 shares_response = instruments_service.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_ALL)

#                 instrument_data_list = []
#                 for i in shares_response.instruments:
#                     instrument_data = {
#                        "name": i.name, "ticker": i.ticker, "uid": i.uid, "figi": i.figi,
#                        "isin": i.isin, "lot": i.lot, "currency": i.currency,
#                        "class_code": i.class_code,
#                        "exchange": i.exchange,
#                        # --- FIX START ---
#                        # The SDK returns datetime objects directly, no need for .ToDatetime()
#                        "first_1min_candle_date": pd.to_datetime(i.first_1min_candle_date) if i.first_1min_candle_date else pd.NaT,
#                        "first_1day_candle_date": pd.to_datetime(i.first_1day_candle_date) if i.first_1day_candle_date else pd.NaT
#                        # --- FIX END ---
#                     }
#                     instrument_data_list.append(instrument_data)

#                 if not instrument_data_list:
#                      self.logger.warning("No instruments retrieved from The Tinkoff API.")
#                      # Keep self.df_screener_all and self.df_screener_rub as empty DataFrames initialized in __init__
#                      return # Exit early

#                 self.df_screener_all = pd.DataFrame(instrument_data_list)
#                 self.df_screener_rub = self.df_screener_all[self.df_screener_all["currency"] == "rub"].copy()
#                 self.logger.info(f"Initialized {len(self.df_screener_all)} instruments, {len(self.df_screener_rub)} in RUB.")

#         except tinkoff.invest.exceptions.RequestError as e:
#              # Catch specific Tinkoff API errors (like authentication)
#              self.logger.error(f"Tinkoff API request error during instrument initialization: {e.code} - {e.details}", exc_info=True)
#         except AttributeError as e:
#              # Catch AttributeErrors like the one reported, but log it specifically
#              self.logger.error(f"Attribute error during instrument processing (check SDK structure?): {e}", exc_info=True)
#         except Exception as e:
#             # Catch any other unexpected errors
#             self.logger.error(f"Unexpected error during instrument initialization: {e}", exc_info=True)
#         finally:
#              # Ensure dataframes are not None if initialization fails partially
#              if self.df_screener_all is None: self.df_screener_all = pd.DataFrame()
#              if self.df_screener_rub is None: self.df_screener_rub = pd.DataFrame()


#     def get_instrument_details(self, ticker: str) -> Optional[Dict]:
#          """Получить FIGI, UID, ISIN и другую инфу для тикера (предпочитая RUB)."""
#          # Check if initialization failed completely
#          if self.df_screener_rub is None or self.df_screener_all is None:
#              self.logger.error("Instrument screener dataframes are None, initialization likely failed.")
#              return None
#          # Check if RUB list is empty, but all list might have it
#          if self.df_screener_rub.empty and self.df_screener_all.empty:
#              self.logger.warning("Instrument lists are empty. Cannot find details.")
#              # Maybe try re-initializing? Or just return None.
#              # self._initialize_instruments() # Be careful with re-init loops
#              return None

#          ticker_info = self.df_screener_rub[self.df_screener_rub["ticker"] == ticker]

#          if ticker_info.empty:
#              self.logger.warning(f"No RUB instrument found for ticker {ticker}. Checking all instruments...")
#              ticker_info_all = self.df_screener_all[self.df_screener_all["ticker"] == ticker]
#              if ticker_info_all.empty:
#                   self.logger.error(f"Ticker {ticker} not found in any instrument list.")
#                   return None
#              else:
#                   self.logger.warning(f"Found ticker {ticker} in non-RUB list. Using first entry.")
#                   instrument_data = ticker_info_all.iloc[0].to_dict()
#                   self.logger.info(f"Selected non-RUB instrument {ticker}: FIGI {instrument_data.get('figi')} ({instrument_data.get('currency')})")
#                   return instrument_data

#          elif ticker_info.shape[0] > 1:
#              self.logger.warning(f"Multiple RUB instruments found for {ticker}. Selecting first entry.")
#              # Add more sophisticated selection logic if needed (e.g., based on class_code like TQBR)
#              instrument_data = ticker_info.iloc[0].to_dict()
#              self.logger.info(f"Selected instrument {ticker}: FIGI {instrument_data.get('figi')} (from multiple RUB options)")
#              return instrument_data
#          else:
#              instrument_data = ticker_info.iloc[0].to_dict()
#              self.logger.info(f"Found unique RUB instrument for {ticker}: FIGI {instrument_data.get('figi')}")
#              return instrument_data

#     def download_missing_raw_years(self, ticker: str, figi: str, years: List[int]) -> List[str]:
#         """
#         Скачать ZIP-файлы сырых данных только для тех лет, которые отсутствуют локально.
#         Возвращает список путей к *новым* скачанным ZIP файлам.
#         """
#         newly_downloaded_zip_paths = []
#         if not figi:
#              self.logger.error(f"Cannot download for {ticker}: FIGI is missing.")
#              return []
#         if not self.token:
#              self.logger.error(f"Cannot download for {ticker}: API token is missing.")
#              return []


#         self.logger.info(f"Checking/Downloading raw data for {ticker} (FIGI: {figi}) for years: {years}")

#         for year in sorted(list(set(years))):
#             if self.storage.check_raw_zip_exists(ticker, figi, year):
#                 self.logger.info(f"Raw data ZIP for {ticker} ({figi}) year {year} already exists. Skipping download.")
#                 continue

#             self.logger.info(f"Attempting download for {ticker} ({figi}) year {year}.")
#             # Note: Tinkoff history API only provides '1min' interval via this endpoint
#             params = {"figi": figi, "year": year}
#             headers = {"Authorization": f"Bearer {self.token}"}
#             retries = 3
#             delay = 5

#             for attempt in range(retries):
#                 try:
#                     response = requests.get(self.url, params=params, headers=headers, timeout=90) # Increased timeout

#                     if response.status_code == 200:
#                         if response.content: # Check if content is not empty
#                             self.logger.info(f"Successfully downloaded data for {ticker} ({figi}) year {year}.")
#                             saved_path = self.storage.store_raw_data(ticker, figi, year, response.content)
#                             if saved_path:
#                                 newly_downloaded_zip_paths.append(saved_path)
#                         else:
#                             self.logger.warning(f"Downloaded data for {ticker} ({figi}) year {year}, but content is empty.")
#                         break # Success

#                     elif response.status_code == 401: # Unauthorized
#                          self.logger.error(f"Authorization failed (401) for {ticker} year {year}. Check API token.")
#                          # No point retrying auth errors usually
#                          return [] # Stop downloads for this ticker if token fails

#                     elif response.status_code == 429: # Rate limit
#                         self.logger.warning(f"Rate limit (429) for {ticker} year {year}. Attempt {attempt + 1}/{retries}. Sleeping {delay}s...")
#                         time.sleep(delay)
#                         delay *= 2

#                     elif response.status_code == 404: # Not found
#                         self.logger.warning(f"No data found (404) for {ticker} ({figi}) year {year}. Likely too early or issue with FIGI/year.")
#                         break # No point retrying a 404

#                     else: # Other HTTP errors
#                         self.logger.error(f"HTTP Error {response.status_code} for {ticker} year {year}: {response.text[:200]}. Attempt {attempt + 1}/{retries}.")
#                         if attempt < retries - 1:
#                              time.sleep(delay) # Wait before retrying other server-side issues
#                         else:
#                              self.logger.error(f"Failed to download {ticker} year {year} after {retries} attempts due to HTTP {response.status_code}.")

#                 except requests.exceptions.Timeout:
#                      self.logger.warning(f"Timeout downloading {ticker} year {year}. Attempt {attempt + 1}/{retries}.")
#                      if attempt < retries - 1:
#                          time.sleep(delay)
#                          delay *= 1.5 # Less aggressive backoff for timeout?
#                      else:
#                          self.logger.error(f"Failed to download {ticker} year {year} due to timeout after {retries} attempts.")

#                 except requests.exceptions.RequestException as e:
#                     self.logger.error(f"Network error downloading {ticker} year {year}: {e}. Attempt {attempt + 1}/{retries}.")
#                     if attempt < retries - 1:
#                         time.sleep(delay)
#                         delay *= 2
#                     else:
#                         self.logger.error(f"Failed to download {ticker} year {year} due to network issues after {retries} attempts.")

#             time.sleep(0.5) # Small courtesy delay between years

#         self.logger.info(f"Download check complete for {ticker}. Newly downloaded: {len(newly_downloaded_zip_paths)} ZIP files.")
#         return newly_downloaded_zip_paths

#     def process_new_raw_data(self, ticker: str, new_zip_files: List[str]) -> Optional[pd.DataFrame]:
#          """
#          Обработать (распаковать, загрузить) только новые ZIP файлы.
#          Возвращает DataFrame с данными *только* из этих новых файлов.
#          """
#          if not new_zip_files:
#               self.logger.info(f"No new raw ZIP files to process for {ticker}.")
#               return None

#          self.log_start_process(f"Processing {len(new_zip_files)} new raw ZIPs for {ticker}")
#          extracted_csvs = self.storage.extract_specific_raw_zips(new_zip_files)
#          new_data_df = self.storage.load_specific_raw_csv_files(extracted_csvs)

#          if new_data_df is not None and not new_data_df.empty:
#             self.logger.info(f"Successfully processed {len(new_data_df)} new rows for {ticker}.")
#          else:
#             self.logger.warning(f"Processing new raw files for {ticker} yielded no data.")

#          # Consider cleanup of extracted CSVs/folders here if desired and deemed safe
#          # E.g., loop through extracted_csvs and remove them and their parent folders if empty

#          self.log_end_process(f"Processing new raw data for {ticker}")
#          return new_data_df


# # ==============================================
# #      MarketDataManager Class (Unchanged)
# # ==============================================
# class MarketDataManager(BaseLogger):
#     def __init__(self, token: str, base_directory: str):
#         super().__init__('MarketDataManager')
#         # Validate token presence early
#         if not token:
#             self.logger.error("MarketDataManager initialized without a valid API token!")
#             # Decide behaviour: raise error, or proceed but expect failures?
#             # raise ValueError("API token is required for MarketDataManager")
#         self.storage = DataStorage(base_directory)
#         self.downloader = TinkoffDataDownloader(token, self.storage) # Pass token here
#         self.converter = TimeframeConverter()
#         self.logger.info("MarketDataManager initialized")


#     def get_data(self, ticker: str, start_date: str, end_date: str, timeframe: str = "1min", force_download: bool = False) -> Optional[pd.DataFrame]:
#         """
#         Получить данные для тикера с указанным временным интервалом,
#         оптимизируя загрузку и обработку.
#         """
#         self.log_start_process(f"Get data: {ticker} ({start_date} to {end_date}), TF={timeframe}, Force={force_download}")

#         try:
#             start_date_dt = pd.to_datetime(start_date)
#             # Make end_date inclusive by setting time to end of day
#             end_date_dt = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59, microsecond=999999)
#         except Exception as e:
#              self.logger.error(f"Invalid date format for {ticker}: {start_date}, {end_date}. Error: {e}")
#              self.log_end_process(f"Get data for {ticker} (failed date parsing)")
#              return None


#         processed_data = None
#         if not force_download:
#             processed_data = self.storage.load_processed_data(ticker)
#             if processed_data is not None and not processed_data.empty:
#                  self.logger.info(f"Loaded {len(processed_data)} rows from existing parquet for {ticker}.")
#             elif processed_data is not None and processed_data.empty:
#                  self.logger.info(f"Existing parquet for {ticker} is empty.")
#             # else: processed_data is None (logged in load_processed_data)
#         else:
#             self.logger.info(f"Force download triggered for {ticker}. Existing parquet check skipped for download decision.")


#         instrument_info = self.downloader.get_instrument_details(ticker)
#         if not instrument_info or not instrument_info.get('figi'):
#              self.logger.error(f"Cannot get FIGI for {ticker}. Aborting data retrieval.")
#              self.log_end_process(f"Get data for {ticker}")
#              return None
#         figi = instrument_info['figi']


#         missing_ranges = []
#         # Determine ranges needed based on force_download or comparison with existing data
#         if force_download or processed_data is None or processed_data.empty:
#              self.logger.info(f"Need full range for {ticker}: {start_date_dt} to {end_date_dt}")
#              missing_ranges = [(start_date_dt, end_date_dt)]
#         else:
#              missing_ranges = self.storage.get_missing_date_ranges(ticker, start_date_dt, end_date_dt)


#         newly_downloaded_zips = []
#         if missing_ranges:
#             required_years = set()
#             for start, end in missing_ranges:
#                 # Ensure start/end are valid before accessing year
#                 if pd.notna(start) and pd.notna(end):
#                      required_years.update(range(start.year, end.year + 1))
#             if required_years:
#                  self.logger.info(f"Years requiring check/download for {ticker}: {sorted(list(required_years))}")
#                  # Pass the confirmed figi to the downloader
#                  newly_downloaded_zips = self.downloader.download_missing_raw_years(ticker, figi, list(required_years))
#             else:
#                  self.logger.info(f"No specific years identified from missing ranges for {ticker}: {missing_ranges}")


#         # Process data only if new zips were downloaded
#         if newly_downloaded_zips:
#              new_data_df = self.downloader.process_new_raw_data(ticker, newly_downloaded_zips)
#              if new_data_df is not None and not new_data_df.empty:
#                   self.logger.info(f"Merging {len(new_data_df)} new rows for {ticker} into parquet.")
#                   self.storage.save_processed_data(ticker, new_data_df)
#                   # Reload full data after merge to ensure consistency
#                   self.logger.info(f"Reloading full processed data for {ticker} after merge.")
#                   processed_data = self.storage.load_processed_data(ticker) # Overwrite potentially outdated var
#              else:
#                   self.logger.warning(f"New ZIPs downloaded for {ticker}, but processing yielded no data. Using previously loaded data (if any).")
#                   # No change to processed_data needed if new data was empty/None
#                   # But if processed_data was None initially, try loading again in case старый файл был создан, но не загружен
#                   if processed_data is None:
#                         processed_data = self.storage.load_processed_data(ticker)


#         # --- Final Filtering and Resampling ---
#         if processed_data is None or processed_data.empty:
#             self.logger.warning(f"No processed data available for {ticker} after update checks.")
#             self.log_end_process(f"Get data for {ticker}")
#             return None

#         # Filter *final* dataset (potentially updated) to the *precise* requested start/end times
#         # Ensure 'date' is datetime before filtering
#         if not pd.api.types.is_datetime64_any_dtype(processed_data['date']):
#              processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
#              processed_data.dropna(subset=['date'], inplace=True)


#         final_data = processed_data[
#              (processed_data['date'] >= start_date_dt) &
#              (processed_data['date'] <= end_date_dt)
#         ].copy()

#         self.logger.info(f"Filtered final data for {ticker} ({start_date_dt} to {end_date_dt}). Rows: {len(final_data)}")


#         # Resample if needed and if data exists after filtering
#         if timeframe != "1min" and not final_data.empty:
#             try:
#                 self.logger.info(f"Resampling {ticker} data from 1min to {timeframe}...")
#                 resampled_data = self.converter.resample_ohlcv(final_data, timeframe)
#                 self.logger.info(f"Resampled {ticker} to {len(resampled_data)} rows at {timeframe}.")
#                 final_data = resampled_data
#             except Exception as e:
#                  self.logger.error(f"Failed to resample data for {ticker} to {timeframe}: {e}", exc_info=True)
#                  # Decide: return 1min data or None? Returning 1min might be confusing.
#                  # Let's return None if resampling fails critically.
#                  self.log_end_process(f"Get data for {ticker} (failed resampling)")
#                  return None
#         elif timeframe == "1min":
#              self.logger.info(f"Timeframe is 1min, no resampling needed for {ticker}.")
#         else: # timeframe != "1min" and final_data is empty
#              self.logger.info(f"No data for {ticker} in the requested range after filtering, cannot resample to {timeframe}.")


#         self._log_data_stats(ticker, final_data, timeframe)
#         self.log_end_process(f"Get data for {ticker}")
#         return final_data

#     def get_data_for_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str, timeframe: str = "1min", max_workers: int = 5) -> Dict[str, Optional[pd.DataFrame]]:
#         """Получить данные для нескольких тикеров с использованием многопоточности."""
#         self.logger.info(f"Starting parallel data retrieval for {len(tickers)} tickers. Workers: {max_workers}")
#         results = {}

#         # Filter out any potentially problematic/empty tickers early?
#         valid_tickers = [t for t in tickers if t and isinstance(t, str)]
#         if len(valid_tickers) != len(tickers):
#             self.logger.warning(f"Removed invalid entries from ticker list. Original: {len(tickers)}, Valid: {len(valid_tickers)}")
#             tickers = valid_tickers
#             if not tickers:
#                  self.logger.error("No valid tickers provided.")
#                  return {}


#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_ticker = {
#                 executor.submit(self.get_data, ticker, start_date, end_date, timeframe): ticker
#                 for ticker in tickers
#             }

#             # Consider using concurrent.futures.as_completed for better progress/error handling
#             for future in future_to_ticker: # Or: for future in concurrent.futures.as_completed(future_to_ticker):
#                 ticker = future_to_ticker[future]
#                 try:
#                     data = future.result() # Retrieve result, potential exceptions raised here
#                     results[ticker] = data
#                     if data is not None and not data.empty:
#                         self.logger.debug(f"Worker finished successfully for {ticker} (Rows: {len(data)})")
#                     elif data is not None and data.empty:
#                         self.logger.debug(f"Worker finished for {ticker}, no data found in range/timeframe.")
#                     else: # data is None
#                          self.logger.warning(f"Worker finished for {ticker}, but retrieval failed (result is None).")

#                 except Exception as e:
#                     # Log the exception from the worker thread
#                     self.logger.error(f"Error in worker thread for ticker {ticker}: {e}", exc_info=True)
#                     results[ticker] = None # Ensure failed tickers have None result

#         self.logger.info(f"Completed parallel data retrieval for {len(tickers)} tickers.")
#         return results

#     def _log_data_stats(self, ticker, data, timeframe):
#         """Вывести статистику о финальных данных."""
#         if data is None or data.empty:
#             self.logger.info(f"{ticker}: No final data available for timeframe {timeframe} in the requested range.")
#             return

#         self.logger.info(f"{ticker} final data stats (Timeframe: {timeframe}, Range: {data['date'].min()} to {data['date'].max()}):")
#         self.logger.info(f"  - Total rows: {len(data)}")
#         # Add more stats if useful, e.g., NaN counts
#         # missing_volume = data['volume'].isnull().sum()
#         # if missing_volume > 0: self.logger.info(f"  - Rows with missing volume: {missing_volume}")


# # ==============================================
# #      MarketDataPipeline Class (Unchanged)
# # ==============================================
# class MarketDataPipeline(BaseLogger):
#     def __init__(self, base_directory=None):
#         super().__init__('MarketDataPipeline')
#         if base_directory is None:
#              # Use BASE_PATH determined globally, assuming it points to parent of 'data'
#               self.base_directory = os.path.join(BASE_PATH, 'data') # Standard structure
#         else:
#              self.base_directory = base_directory

#         try:
#              os.makedirs(self.base_directory, exist_ok=True)
#              self.logger.info(f"MarketDataPipeline initialized. Base directory: {self.base_directory}")
#         except OSError as e:
#              self.logger.error(f"Failed to create base directory {self.base_directory}: {e}")
#              # Decide how to handle this - raise error? Set directory to None?
#              raise


#     def run(self, tickers: List[str], start_date: str, end_date: str, token: str, timeframe: str = "1min", max_workers: int = 5) -> Dict[str, Optional[pd.DataFrame]]:
#         """
#         Запускает пайплайн загрузки данных.
#         """
#         self.logger.info(f"Starting market data pipeline for {len(tickers)} tickers...")
#         self.logger.info(f"Date range: {start_date} to {end_date}, Timeframe: {timeframe}, Workers: {max_workers}")

#         if not token:
#              self.logger.error("Tinkoff API token is missing or invalid. Cannot proceed.")
#              return {ticker: None for ticker in tickers}

#         if not tickers:
#              self.logger.warning("Ticker list is empty. Nothing to run.")
#              return {}

#         try:
#             manager = MarketDataManager(
#                  token=token,
#                  base_directory=self.base_directory
#             )
#         except Exception as e:
#              self.logger.error(f"Failed to initialize MarketDataManager: {e}", exc_info=True)
#              return {ticker: None for ticker in tickers} # Return None for all if manager fails

#         self.log_start_process("Data download and processing (parallel)")
#         try:
#             data = manager.get_data_for_multiple_tickers(
#                 tickers=tickers,
#                 start_date=start_date,
#                 end_date=end_date,
#                 timeframe=timeframe,
#                 max_workers=max_workers
#             )
#         except Exception as e:
#             # Catch errors potentially raised from within get_data_for_multiple_tickers itself
#             # (though most errors should be caught within the loop or get_data)
#             self.logger.error(f"Critical error during parallel data retrieval: {e}", exc_info=True)
#             # Ensure all tickers have a None result if the main call fails
#             data = {ticker: None for ticker in tickers}

#         self.log_end_process("Data download and processing (parallel)")

#         # Summarize results
#         success_count = sum(1 for df in data.values() if df is not None and not df.empty)
#         empty_count = sum(1 for df in data.values() if df is not None and df.empty)
#         fail_count = sum(1 for df in data.values() if df is None)
#         total_processed = len(data) # Should match len(tickers)

#         self.logger.info(f"Pipeline finished. Processed: {total_processed}/{len(tickers)} tickers.")
#         self.logger.info(f"  Success (data found): {success_count}")
#         self.logger.info(f"  Success (no data in range): {empty_count}")
#         self.logger.info(f"  Failed: {fail_count}")

#         if fail_count > 0:
#              failed_tickers = [ticker for ticker, df in data.items() if df is None]
#              self.logger.warning(f"Tickers with failed retrieval: {failed_tickers}")

#         return data

# # ==============================================
# #      Helper Function run_pipeline_market
# # ==============================================
# def run_pipeline_market(tickers: List[str], start_date: str, end_date: str, token: str, timeframe: str="1min", max_workers: int = 5, base_directory: Optional[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
#     """
#     Helper function to instantiate and run the MarketDataPipeline.

#     Args:
#         tickers: List of ticker symbols.
#         start_date: Start date string (YYYY-MM-DD).
#         end_date: End date string (YYYY-MM-DD).
#         token: Tinkoff API token.
#         timeframe: Desired timeframe ('1min', '5min', '1h', '1d', etc.).
#         max_workers: Max threads for parallel download.
#         base_directory: Optional override for the data storage directory.
#                         If None, uses the default logic (BASE_PATH + '/data').

#     Returns:
#          Dictionary mapping ticker symbols to pandas DataFrames (or None on failure).
#     """
#     # Use the global BASE_PATH if base_directory is not provided to the function
#     effective_base_directory = base_directory
#     if effective_base_directory is None:
#          # Construct the default path dynamically when the function is called
#           try:
#                 # Re-check BASE_PATH validity just in case
#                 if 'BASE_PATH' not in globals() or not BASE_PATH:
#                       print("Error: Global BASE_PATH is not defined or empty.")
#                       # Attempt fallback to determine project root again if necessary
#                       # This is getting complex, ideally BASE_PATH is set correctly once globally
#                       # Fallback: Use current working directory + /data? Risky.
#                       effective_base_directory = os.path.join(os.getcwd(), 'data')
#                       print(f"Warning: Using fallback data directory: {effective_base_directory}")
#                 else:
#                       effective_base_directory = os.path.join(BASE_PATH, 'data')
#           except Exception as e:
#                print(f"Error determining default base directory: {e}")
#                raise ValueError("Could not determine a valid base directory for data.") from e


#     try:
#         pipeline = MarketDataPipeline(base_directory=effective_base_directory)
#         return pipeline.run(tickers, start_date, end_date, token, timeframe, max_workers)
#     except Exception as e:
#          # Catch initialization error of the pipeline itself (e.g., directory creation fails)
#          print(f"Error creating or running MarketDataPipeline: {e}")
#          # Return empty dict or dict with None for all tickers?
#          return {ticker: None for ticker in tickers}

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
    
sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection')
# # from private_info import BASE_PATH
# token = 't.6vI7rHdW8TEcg9R6KlKyOeHVNyGnPdlyIwkaK5DbJxGdOI_PM7UIPJJpdjpYkHP7GvLWvkiVnX_mXWMWAPYJnQ'
# YOUR_API_ID = 28994010
# YOUR_API_HASH = '6e7a57fdfe1a10b0a3434104b42badf2'
# BASE_PATH = '/Users/bolotnikovali/WORKING/kursach'

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
