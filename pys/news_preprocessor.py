import pandas as pd
import re
import os
import glob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class NewsPreprocessor:
    """Класс для очистки и предобработки новостных данных"""
    
    def __init__(self, base_dir, use_nltk=True):
        """
        Инициализация предобработчика новостей
        
        Args:
            base_dir (str): Базовая директория с данными
            use_nltk (bool): Использовать ли NLTK для обработки текста
        """
        self.base_dir = base_dir
        self.use_nltk = use_nltk
        
        if use_nltk:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
    
    def clean_text(self, text):
        """
        Очистка текста новостей от специальных символов, ссылок и emoji
        
        Args:
            text (str): Исходный текст новости
            
        Returns:
            str: Очищенный текст
        """
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'[^\w\s,.!?;:()\'\"-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_news_file(self, file_path, save=True):
        """
        Обработка одного файла с новостями
        
        Args:
            file_path (str): Путь к файлу с новостями
            save (bool): Сохранять ли результат в файл
            
        Returns:
            pd.DataFrame: DataFrame с обработанными новостями
        """
        try:
            df = pd.read_csv(file_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            df['clean_text'] = df['text'].apply(self.clean_text)
            
            df = df[df['clean_text'].str.len() > 20]
            
            if save:
                dir_path = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                

                processed_file_name = file_name.replace('.csv', '_processed.csv')
                processed_file_path = os.path.join(dir_path, processed_file_name)
                
                df.to_csv(processed_file_path, index=False)
                print(f"Обработанные данные сохранены в {processed_file_path}")
            
            return df
        
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            return pd.DataFrame()
    
    def process_ticker_news(self, ticker, save=True):
        """
        Обработка всех файлов новостей для конкретного тикера
        
        Args:
            ticker (str): Тикер компании
            save (bool): Сохранять ли результат в файл
            
        Returns:
            pd.DataFrame: DataFrame со всеми обработанными новостями для тикера
        """
        ticker_dir = os.path.join(self.base_dir, 'data', 'processed_data', ticker)
        
        if not os.path.exists(ticker_dir):
            print(f"Директория {ticker_dir} не существует")
            return pd.DataFrame()
        
        news_files = glob.glob(os.path.join(ticker_dir, f"{ticker}_news_*.csv"))
        
        if not news_files:
            print(f"Не найдено файлов новостей для тикера {ticker}")
            return pd.DataFrame()
        
        processed_dfs = []
        
        for file_path in news_files:
            df = self.process_news_file(file_path, save=False)
            if not df.empty:
                processed_dfs.append(df)
        
        if not processed_dfs:
            print(f"Не удалось обработать файлы новостей для тикера {ticker}")
            return pd.DataFrame()
        
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date', 'clean_text'])
        combined_df = combined_df.sort_values('date')
        
        if save:
            combined_file_path = os.path.join(ticker_dir, f"{ticker}_all_news_processed.csv")
            combined_df.to_csv(combined_file_path, index=False)
            print(f"Объединенные данные сохранены в {combined_file_path}")
        
        return combined_df
    
    def process_all_tickers(self, tickers=None, save=True):
        """
        Обработка новостей для всех указанных тикеров
        
        Args:
            tickers (list): Список тикеров. Если None, будут обработаны все тикеры в директории
            save (bool): Сохранять ли результат в файл
            
        Returns:
            dict: Словарь {ticker: DataFrame} с обработанными новостями для каждого тикера
        """

        if tickers is None:
            processed_data_dir = os.path.join(self.base_dir, 'processed_data')
            tickers = [d for d in os.listdir(processed_data_dir) 
                       if os.path.isdir(os.path.join(processed_data_dir, d))]
        
        processed_news = {}
        
        for ticker in tickers:
            print(f"Обработка новостей для тикера {ticker}...")
            df = self.process_ticker_news(ticker, save=save)
            if not df.empty:
                processed_news[ticker] = df
        
        return processed_news
