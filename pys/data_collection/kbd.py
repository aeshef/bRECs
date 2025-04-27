import requests
import pandas as pd
import os
import traceback
from datetime import datetime, timedelta

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class KBDDownloader(BaseLogger):
    """
    Класс для загрузки данных кривой бескупонной доходности (КБД) с MOEX API
    """
    
    def __init__(self, output_dir=f'{BASE_PATH}/data/processed_data/BONDS/kbd'):
        """
        Инициализация загрузчика данных КБД
        
        :param output_dir: Директория для сохранения данных
        """
        super().__init__('KBDDownloader')
        # Гарантируем, что все данные сохраняются в указанной директории
        self.output_dir = output_dir
        
        # Организуем логическую структуру подпапок
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.raw_dir = os.path.join(self.output_dir, 'raw')
        
        # Создаем нужные директории
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        self.moex_url = "https://iss.moex.com/iss/apps/bondization/zcyc_range_calculator.csv"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        self.logger.info("KBDDownloader initialized")
    
    def get_kbd(self, start_date, end_date):
        """
        Получить данные КБД с MOEX API за указанный период и сохранить их в CSV
        
        :param start_date: Начальная дата в формате datetime
        :param end_date: Конечная дата в формате datetime
        :return: DataFrame с данными КБД или None в случае ошибки
        """
        self.logger.info(f"Fetching KBD data from {start_date} to {end_date}")
        
        try:
            # Форматируем даты в нужный формат для API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Формируем параметры запроса
            params = {
                'from': start_str,
                'till': end_str,
                'periods': '0.25,0.5,0.75,1,2,3,5,7,10,15,20,30',  # Тенора
                'iss.dp': 'comma',  # Использовать запятую как десятичный разделитель
                'iss.df': '%d.%m.%Y'  # Формат даты (без экранирования)
            }
            
            # Отправляем запрос
            response = requests.get(self.moex_url, params=params, headers=self.headers)
            
            if response.status_code != 200:
                self.logger.error(f"MOEX API request error: HTTP {response.status_code}")
                return None
            
            # Сохраняем сырые данные для анализа
            raw_file_path = os.path.join(self.raw_dir, f'raw_kbd_data_{start_str}_to_{end_str}.csv')
            with open(raw_file_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Raw KBD data saved to {raw_file_path}")
            
            # Парсим CSV данные
            df = self._parse_moex_csv(raw_file_path)
            
            if df is not None and not df.empty:
                # Сохраняем обработанные данные в основной файл
                file_path = os.path.join(self.data_dir, 'kbd_data.csv')
                df.to_csv(file_path, index=False)
                self.logger.info(f"Processed KBD data saved to {file_path} with {len(df)} rows")
                
                # Сохраняем также копию с датой получения
                dated_file_path = os.path.join(self.data_dir, f'kbd_data_{datetime.now().strftime("%Y%m%d")}.csv')
                df.to_csv(dated_file_path, index=False)
                
                return df
            else:
                self.logger.error("Failed to parse MOEX data")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception during KBD data retrieval: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _parse_moex_csv(self, file_path):
        """
        Парсинг CSV-файла от MOEX API
        
        :param file_path: Путь к CSV-файлу
        :return: DataFrame с данными КБД
        """
        try:
            # Проверяем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.logger.info(f"Raw file contains {len(lines)} lines")
                
                # Определяем, есть ли строка "zcyc" в начале
                first_line_is_zcyc = lines[0].strip() == 'zcyc'
                skip_rows = 1 if first_line_is_zcyc else 0
                
                if first_line_is_zcyc:
                    self.logger.info("First line contains 'zcyc', skipping it")
            
            # Читаем CSV с правильным числом пропускаемых строк
            df = pd.read_csv(
                file_path, 
                sep=';', 
                skiprows=skip_rows,
                encoding='utf-8'
            )
            
            self.logger.info(f"Parsed CSV with columns: {df.columns.tolist()}")
            
            # Преобразуем в нужный формат для дальнейшего анализа
            result_df = pd.DataFrame()
            
            # Обрабатываем дату
            if 'tradedate' in df.columns:
                try:
                    result_df['date'] = pd.to_datetime(df['tradedate'], format='%d.%m.%Y')
                    self.logger.info(f"Successfully parsed dates, first date: {result_df['date'].iloc[0]}")
                except Exception as e:
                    self.logger.error(f"Error parsing dates: {e}")
                    return None
            else:
                self.logger.error("Missing tradedate column in MOEX data")
                return None
            
            # Маппинг колонок периодов в стандартный формат
            period_mapping = {
                'period_0.25': '0.25Y',
                'period_0.5': '0.5Y',
                'period_0.75': '0.75Y',
                'period_1.0': '1Y',
                'period_2.0': '2Y',
                'period_3.0': '3Y',
                'period_5.0': '5Y',
                'period_7.0': '7Y',
                'period_10.0': '10Y',
                'period_15.0': '15Y',
                'period_20.0': '20Y',
                'period_30.0': '30Y'
            }
            
            # Копируем и преобразуем колонки с периодами
            for moex_col, std_col in period_mapping.items():
                if moex_col in df.columns:
                    # Заменяем запятые на точки и преобразуем в числа
                    result_df[std_col] = pd.to_numeric(df[moex_col].astype(str).str.replace(',', '.'), errors='coerce')
            
            self.logger.info(f"Processed MOEX data: {len(result_df)} rows with {len([c for c in result_df.columns if c != 'date'])} tenors")
            return result_df
                
        except Exception as e:
            self.logger.error(f"Error parsing MOEX CSV: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
def run_pipeline_kbd_parser(base_path=BASE_PATH, start_date=None, end_date=None, update_data=True):
    """
    Загружает данные КБД и возвращает их в виде DataFrame
    
    Parameters:
    -----------
    base_path : str
        Базовый путь проекта
    start_date : datetime или str
        Начальная дата (если None, используется год назад)
    end_date : datetime или str
        Конечная дата (если None, используется текущая дата)
    update_data : bool
        Обновлять ли данные с сервера MOEX
    
    Returns:
    --------
    pandas.DataFrame
        Данные КБД или None в случае ошибки
    """
    from datetime import datetime, timedelta
    import os
    
    # Настройка директорий
    kbd_dir = f"{base_path}/data/processed_data/BONDS/kbd"
    
    # Установка дат для КБД, если не указаны
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Инициализируем загрузчик
    downloader = KBDDownloader(output_dir=kbd_dir)
    
    # Загружаем данные КБД
    kbd_data = None
    
    if update_data:
        # Загружаем актуальные данные с MOEX API
        kbd_data = downloader.get_kbd(start_date, end_date)
        
        if kbd_data is None or kbd_data.empty:
            kbd_data = downloader.load_kbd_data()
    else:
        kbd_data = downloader.load_kbd_data()
    
    return kbd_data
