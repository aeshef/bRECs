import requests
import pandas as pd
import os
import sys
from bs4 import BeautifulSoup
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir):
        break

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.logger import BaseLogger

class KBDDownloader(BaseLogger):
    """
    Класс для загрузки данных кривой бескупонной доходности (КБД) с сайта ЦБ РФ
    """
    
    def __init__(self, output_dir='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
        """
        Инициализация загрузчика данных КБД
        
        :param output_dir: Директория для сохранения данных
        """
        super().__init__('KBDDownloader')
        self.output_dir = output_dir
        self.url = "https://cbr.ru/hd_base/zcyc_params/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        self.logger.info("KBDDownloader initialized")
        
    def get_kbd(self, start_date, end_date):
        """
        Получить данные КБД за указанный период и сохранить их в CSV
        
        :param start_date: Начальная дата в формате datetime
        :param end_date: Конечная дата в формате datetime
        :return: DataFrame с данными КБД или None в случае ошибки
        """
        self.logger.info(f"Fetching KBD data from {start_date} to {end_date}")
        
        session = requests.Session()
        session.headers.update(self.headers)

        try:
            response = session.get(self.url)
            
            if response.status_code == 200:
                self.logger.info("Successfully retrieved page from CBR")
                soup = BeautifulSoup(response.content, 'html.parser')
                
                table = soup.find('table', {'class': 'data spaced'})
                
                if table:
                    # Извлекаем заголовки таблицы
                    headers = []
                    header_row = table.find_all('tr')[1]
                    header_columns = header_row.find_all('th')[1:]
                    for col in header_columns:
                        headers.append(col.text.strip())
                    
                    self.logger.debug(f"Found table headers: {headers}")
                    
                    # Извлекаем данные из строк таблицы
                    rows = table.find_all('tr')[2:]
                    data = []
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) > 1:
                            date_str = cols[0].text.strip()
                            date_obj = datetime.strptime(date_str, '%d.%m.%Y')
                            
                            # Фильтруем данные по дате
                            if start_date <= date_obj <= end_date:
                                row_data = {'date': date_obj}
                                
                                for i, col in enumerate(cols[1:]):
                                    if i < len(headers):
                                        row_data[headers[i]] = col.text.strip()
                                
                                data.append(row_data)
                    
                    self.logger.info(f"Extracted {len(data)} rows of KBD data within date range")
                    
                    if data:
                        df = pd.DataFrame(data)
                        
                        # Создаем директорию, если она не существует
                        os.makedirs(self.output_dir, exist_ok=True)
                        file_path = os.path.join(self.output_dir, 'kbd.csv')
                        
                        df.to_csv(file_path, index=False)
                        self.logger.info(f"KBD data successfully saved to {file_path}")
                        
                        return df
                    else:
                        self.logger.warning("No data found within the specified date range")
                        return None
                else:
                    self.logger.error("Table not found on the page")
                    return None
            else:
                self.logger.error(f"Request error: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception during KBD data retrieval: {e}")
            return None
            
    def load_kbd_data(self, file_path=None):
        """
        Загрузить ранее сохраненные данные КБД из CSV файла
        
        :param file_path: Путь к файлу CSV с данными КБД (если None, используется стандартный путь)
        :return: DataFrame с данными КБД
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'kbd.csv')
            
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                self.logger.info(f"Successfully loaded KBD data from {file_path} ({len(df)} rows)")
                return df
            else:
                self.logger.warning(f"KBD data file not found at {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading KBD data: {e}")
            return None
