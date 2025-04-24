import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import time
import logging
import traceback
import argparse
import json
from typing import List, Dict, Optional, Any

from telethon.sync import TelegramClient

sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection/news_processor')

from news_processor.news_preprocessor import NewsPreprocessor
from news_processor.sentiment_analysis import SentimentAnalyzer
from news_processor.news_feature_extractor import NewsFeatureExtractor
from news_processor.event_detector import EventDetector
from news_processor.news_visualizer import NewsVisualizer
from news_processor.news_integration import NewsIntegration

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
    current_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.logger import BaseLogger

import warnings
warnings.filterwarnings('ignore')

def merge_with_existing(new_df: pd.DataFrame, file_path: str, unique_columns: list):
    """
    Загружает существующий CSV, объединяет с new_df и удаляет дубликаты на основе уникальных колонок.
    Если файла нет, возвращает new_df.
    """
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, parse_dates=['date'], encoding='utf-8')
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=unique_columns, keep='last', inplace=True)
            logging.info(f"Объединено данных из {file_path}: {len(combined_df)} записей после объединения")
            return combined_df
        except Exception as e:
            logging.error(f"Ошибка при объединении данных из {file_path}: {e}")
            return new_df
    else:
        return new_df


def save_sentiment_data(sentiment_analyzer, news_df: pd.DataFrame, sentiment_file: str):
    """
    Производит сентимент-анализ новостей с объединением новых и исторических данных.
    Если файл уже существует, выбираются только новые записи, затем объединяются и сохраняются.
    """
    if os.path.exists(sentiment_file):
        logging.info(f"Существующий файл сентимента найден: {sentiment_file}")
        try:
            existing_sentiment = pd.read_csv(sentiment_file, parse_dates=['date'], encoding='utf-8')
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла {sentiment_file}: {e}")
            existing_sentiment = pd.DataFrame()

        # Выборка новых новостей, которых ещё нет в исторических данных.
        if 'id' in existing_sentiment.columns and 'id' in news_df.columns:
            processed_ids = set(existing_sentiment['id'])
            news_to_process = news_df[~news_df['id'].isin(processed_ids)]
        else:
            # Генерируем уникальный ключ, если отсутствует идентификатор.
            existing_sentiment['key'] = existing_sentiment.apply(
                lambda row: f"{str(row.get('date', ''))}_{row.get('clean_text', '')[:50]}",
                axis=1
            )
            news_df['key'] = news_df.apply(
                lambda row: f"{str(row.get('date', ''))}_{row.get('clean_text', '')[:50]}",
                axis=1
            )
            existing_keys = set(existing_sentiment['key'])
            news_to_process = news_df[~news_df['key'].isin(existing_keys)]

        logging.info(f"Найдено {len(news_to_process)} новых новостей для сентимент-анализа")
        sentiment_new = sentiment_analyzer.analyze_ticker_news(news_to_process, save_path=None)
        combined_sentiment = pd.concat([existing_sentiment, sentiment_new], ignore_index=True)
        if 'id' in combined_sentiment.columns:
            combined_sentiment.drop_duplicates(subset='id', keep='last', inplace=True)
        else:
            combined_sentiment.drop_duplicates(subset=['date', 'clean_text'], keep='last', inplace=True)
    else:
        combined_sentiment = sentiment_analyzer.analyze_ticker_news(news_df, save_path=None)

    combined_sentiment.to_csv(sentiment_file, index=False, encoding='utf-8')
    logging.info(f"Сохранено {len(combined_sentiment)} новостей с сентиментом в {sentiment_file}")
    return combined_sentiment


def update_daily_sentiment(sentiment_analyzer, combined_sentiment: pd.DataFrame, output_dir: str, ticker: str):
    """
    Создает или обновляет CSV-файл с ежедневным сентиментом путем объединения новых и исторических данных.
    """
    daily_sentiment_new = sentiment_analyzer.create_daily_sentiment_series(combined_sentiment)
    daily_sentiment_path = os.path.join(output_dir, f"{ticker}_daily_sentiment.csv")
    daily_sentiment = merge_with_existing(daily_sentiment_new, daily_sentiment_path, unique_columns=['date'])
    daily_sentiment.to_csv(daily_sentiment_path, index=False, encoding='utf-8')
    logging.info(f"Ежедневный сентимент обновлен и сохранен в {daily_sentiment_path}")
    return daily_sentiment


def save_events_data(event_detector, combined_sentiment: pd.DataFrame, events_file: str):
    """
    Обнаруживает и оценивает события на основе сентиментированных новостей, затем объединяет
    их с историческими данными и сохраняет результат.
    """
    news_with_events_new = event_detector.detect_events(combined_sentiment)
    news_with_events_new = event_detector.assess_event_impact(news_with_events_new)
    
    unique_columns = ['id'] if 'id' in news_with_events_new.columns else ['date', 'clean_text']
    news_with_events = merge_with_existing(news_with_events_new, events_file, unique_columns=unique_columns)
    news_with_events.to_csv(events_file, index=False, encoding='utf-8')
    logging.info(f"Данные по событиям объединены и сохранены в {events_file}")
    return news_with_events

class NewsPipeline(BaseLogger):
    """Единый пайплайн для сбора и анализа новостей"""
    
    def __init__(self):
        self.telegram_data = {}
        super().__init__('NewsPipeline')
        self.base_dir='/Users/aeshef/Documents/GitHub/kursach'
        
    # def _setup_logger(self):
    #     """Настройка логгера"""
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S'
    #     )
    #     return logging.getLogger('NewsPipeline')

    def _collect_telegram_data(self, api_id, api_hash, channel, limit, output_dir, tickers, start_date=None, end_date=None):
        if not api_id or not api_hash:
            self.logger.warning("API данные не предоставлены. Пропускаем сбор данных Telegram.")
            return {}
            
        os.makedirs(output_dir, exist_ok=True)

        COMPANY_INFO = {
            'SBER': {'name': 'Сбербанк', 'industry': 'банк', 
                    'keywords': ['сбербанк', 'греф', 'сбер', 'банковские услуги']},
            'GAZP': {'name': 'Газпром', 'industry': 'нефтегаз', 
                    'keywords': ['газпром', 'газ', 'миллер', 'энергетика']},
            'LKOH': {'name': 'Лукойл', 'industry': 'нефтегаз', 
                    'keywords': ['лукойл', 'нефть', 'вазов', 'энергетика']},
            'GMKN': {'name': 'ГМК "Норильский никель"', 'industry': 'металлургия', 
                    'keywords': ['норильский никель', 'никель', 'потанин', 'металлы']},
            'ROSN': {'name': 'Роснефть', 'industry': 'нефтегаз', 
                    'keywords': ['роснефть', 'нефть', 'сечин', 'энергетика']},
            'TATN': {'name': 'Татнефть', 'industry': 'нефтегаз', 
                    'keywords': ['татнефть', 'нефть', 'татарыстан', 'энергетика']},
            'MTSS': {'name': 'МТС', 'industry': 'телекоммуникации', 
                    'keywords': ['мтс', 'телеком', 'связь', 'мобильные услуги']},
            'ALRS': {'name': 'АК Алроса', 'industry': 'алмазы', 
                    'keywords': ['алроса', 'алмаз', 'добыча алмазов', 'якутия']},
            'SNGS': {'name': 'Сургутнефтегаз', 'industry': 'нефтегаз', 
                    'keywords': ['сургутнефтегаз', 'нефть', 'сургут', 'энергетика']},
            'VTBR': {'name': 'ВТБ', 'industry': 'банк', 
                    'keywords': ['втб', 'костин', 'сбербанк', 'банковские услуги']},
            'NVTK': {'name': 'Новатэк', 'industry': 'нефтегаз', 
                    'keywords': ['новатэк', 'газ', 'нефть', 'михельсон']},
            'MVID': {'name': 'М.Видео', 'industry': 'розничная торговля', 
                    'keywords': ['м.видео', 'электроника', 'розница', 'техника']},
            'PHOR': {'name': 'ФосАгро', 'industry': 'химическая промышленность', 
                    'keywords': ['фосагро', 'удобрения', 'химия', 'промышленность']},
            'SIBN': {'name': 'Сибнефть', 'industry': 'нефтегаз', 
                    'keywords': ['сибнефть', 'нефть', 'газпром', 'энергетика']},
            'AFKS': {'name': 'АФК "Система"', 'industry': 'конгломерат', 
                    'keywords': ['афк система', 'система', 'евтушенков']},
            'MAGN': {'name': 'Магнитогорский металлургический комбинат', 'industry': 'металлургия', 
                    'keywords': ['магнитогорский', 'металлургия', 'сталь', 'магнитка']},
            'RUAL': {'name': 'Русал', 'industry': 'металлургия', 
                    'keywords': ['русал', 'алюминий', 'дерипаска', 'металлы']}
        }

        results = {}
        try:
            session_file = os.path.join(self.base_dir, 'telegram_session')
            with TelegramClient(session_file, api_id, api_hash) as client:
                print("Подключение к Telegram...", flush=True)
                
                try:
                    entity = client.get_entity(channel)
                    self.logger.info(f"Получена сущность канала: {entity.title}")
                except Exception as e:
                    self.logger.error(f"Ошибка получения сущности: {str(e)}")
                    return {}
                
                # Собираем сообщения
                messages = []
                try:
                    for message in client.iter_messages(entity, limit=limit):
                        msg_date = message.date.date()
                        
                        if start_date and msg_date < start_date:
                            continue
                        if end_date and msg_date > end_date:
                            continue
                        
                        messages.append(message)
                        
                        if len(messages) % 50 == 0:
                            self.logger.info(f"Получено {len(messages)} сообщений...")
                            time.sleep(0.5)
                    
                    self.logger.info(f"Всего получено {len(messages)} сообщений")
                except Exception as e:
                    self.logger.error(f"Ошибка при сборе сообщений: {str(e)}")
                    if not messages:
                        return {}
                
                # Обработка сообщений
                message_data = []
                for msg in messages:
                    if not msg.text:
                        continue
                    
                    # Поиск тикеров в тегах
                    ticker_pattern = r'#([A-Z0-9]{4,6})'
                    tickers_from_tags = [t for t in re.findall(ticker_pattern, msg.text) 
                                        if t in COMPANY_INFO]
                    
                    # Поиск тикеров по ключевым словам
                    tickers_from_keywords = []
                    text_lower = msg.text.lower()
                    for ticker, info in COMPANY_INFO.items():
                        if info['name'].lower() in text_lower:
                            tickers_from_keywords.append(ticker)
                            continue
                        
                        for keyword in info['keywords']:
                            if keyword in text_lower:
                                tickers_from_keywords.append(ticker)
                                break
                    
                    all_tickers = list(set(tickers_from_tags + tickers_from_keywords))
                    if not all_tickers:  # Если нет тикеров, просто пропускаем
                        continue
                        
                    message_data.append({
                        'id': msg.id,
                        'date': msg.date,
                        'text': msg.text,
                        'tickers': all_tickers,
                        'tickers_from_tags': tickers_from_tags,
                        'tickers_from_keywords': tickers_from_keywords,
                        'has_media': msg.media is not None
                    })
                
                # Проверка наличия данных
                if not message_data:
                    self.logger.warning("Не найдено сообщений с тикерами")
                    return {}
                    
                # Создаем DataFrame и сохраняем
                df = pd.DataFrame(message_data)
                
                # Проверим все необходимые колонки
                required_columns = ['id', 'date', 'text', 'tickers', 'tickers_from_tags', 'tickers_from_keywords']
                for col in required_columns:
                    if col not in df.columns:
                        self.logger.error(f"В DataFrame отсутствует необходимая колонка: {col}")
                        if col == 'tickers':
                            df['tickers'] = [[]] * len(df)
                            
                if not df.empty:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    all_messages_file = os.path.join(output_dir, f"telegram_headlines_{timestamp}.csv")
                    df.to_csv(all_messages_file, index=False, encoding='utf-8')
                    self.logger.info(f"Сохранено {len(df)} сообщений в файл {all_messages_file}")
                
                # Разделение по тикерам
                for ticker in tickers:
                    if ticker not in COMPANY_INFO:
                        continue
                        
                    # Проверяем, существуют ли необходимые колонки
                    ticker_messages = df[df['tickers'].apply(
                        lambda x: isinstance(x, list) and ticker in x
                    )].copy() if 'tickers' in df.columns else pd.DataFrame()
                    
                    if ticker_messages.empty:
                        continue
                    
                    # Добавляем информацию
                    ticker_messages['ticker'] = ticker
                    ticker_messages['company_name'] = COMPANY_INFO[ticker]['name']
                    ticker_messages['industry'] = COMPANY_INFO[ticker]['industry']
                    ticker_messages['news_type'] = ticker_messages.apply(
                        lambda row: 'company_specific' if 'tickers_from_tags' in row and ticker in row['tickers_from_tags']
                        else 'industry', axis=1
                    )
                    
                    results[ticker] = ticker_messages
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d")
                    ticker_path = os.path.join(output_dir, f"{ticker}_telegram_news_{timestamp}.csv")
                    ticker_messages.to_csv(ticker_path, index=False, encoding='utf-8')
                    
                self.logger.info(f"Данные разделены и сохранены для {len(results)} тикеров")
                
        except Exception as e:
            self.logger.error(f"Ошибка при сборе Telegram данных: {str(e)}")
            traceback.print_exc()
            
        return results


    def _load_telegram_data(
        self,
        data_dir: str,
        tickers: List[str],
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None
    ) -> Dict[str, pd.DataFrame]:
        """Загрузка кэшированных Telegram данных"""
        results = {}
        
        # Поиск всех файлов с новостями из телеграм
        for ticker in tickers:
            # Ищем в директории telegram_news
            telegram_dir = os.path.join(data_dir, 'telegram_news')
            if os.path.exists(telegram_dir):
                ticker_files = glob.glob(os.path.join(telegram_dir, f"{ticker}_telegram_news_*.csv"))
                
                # Ищем в директории processed_data/{ticker}
                ticker_dir = os.path.join(data_dir, 'processed_data', ticker)
                if os.path.exists(ticker_dir):
                    ticker_files.extend(glob.glob(os.path.join(ticker_dir, f"{ticker}_news_*.csv")))
                    ticker_files.extend(glob.glob(os.path.join(ticker_dir, f"{ticker}_telegram_*.csv")))
                
                if not ticker_files:
                    continue
                    
                # Берем самый свежий файл
                latest_file = max(ticker_files, key=os.path.getmtime)
                
                try:
                    df = pd.read_csv(latest_file, encoding='utf-8')
                    
                    # Преобразуем строковые списки в списки Python
                    # Улучшенная обработка строковых списков
                    for col in ['tickers', 'tickers_from_tags', 'tickers_from_keywords']:
                        if col in df.columns:
                            df[col] = df[col].apply(
                                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and pd.notna(x) else 
                                        ([] if pd.isna(x) else x)
        )
                    
                    # Конвертируем дату
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Применяем фильтр по датам
                        if start_date:
                            df = df[df['date'].dt.date >= start_date]
                        if end_date:
                            df = df[df['date'].dt.date <= end_date]
                    
                    # Удаляем дубликаты
                    if 'id' in df.columns:
                        df = df.drop_duplicates(subset='id')
                        
                    if not df.empty:
                        results[ticker] = df
                        self.logger.info(f"Загружено {len(df)} сообщений для {ticker} из {os.path.basename(latest_file)}")
                        
                except Exception as e:
                    self.logger.error(f"Ошибка при загрузке {latest_file}: {str(e)}")
        
        return results

    def _process_ticker(
        self, 
        ticker: str,
        base_dir: str,
        preprocessor: NewsPreprocessor,
        sentiment_analyzer: SentimentAnalyzer,
        event_detector: EventDetector,
        feature_extractor: NewsFeatureExtractor,
        integrator: NewsIntegration
    ) -> Dict[str, Any]:
        """Обработка одного тикера целиком"""
        results = {}
        output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Шаг 1: Предобработка объединяет все новости, включая новые
        self.logger.info(f"Предобработка новостей для {ticker}")
        news_df = preprocessor.process_ticker_news(ticker, save=True)
        
        if news_df.empty:
            self.logger.warning(f"Нет новостей для {ticker}")
            return results
            
        results['processed_news'] = news_df
        
        # Проверяем, есть ли уже обработанные новости с сентиментом
        sentiment_file = os.path.join(output_dir, f"{ticker}_news_with_sentiment.csv")
        
        # Если файл существует, загружаем и объединяем с новыми данными
        if os.path.exists(sentiment_file):
            self.logger.info(f"Найден существующий файл с сентиментом: {sentiment_file}")
            existing_sentiment = pd.read_csv(sentiment_file)
            
            if 'date' in existing_sentiment.columns:
                existing_sentiment['date'] = pd.to_datetime(existing_sentiment['date'])
            
            # Определяем, какие новости уже обработаны (по id или по тексту и дате)
            if 'id' in existing_sentiment.columns and 'id' in news_df.columns:
                processed_ids = set(existing_sentiment['id'])
                news_to_process = news_df[~news_df['id'].isin(processed_ids)]
            else:
                # Альтернативный метод, если нет id
                existing_sentiment['key'] = existing_sentiment.apply(
                    lambda row: f"{str(row.get('date', ''))}_{row.get('clean_text', '')[:50]}", axis=1
                )
                news_df['key'] = news_df.apply(
                    lambda row: f"{str(row.get('date', ''))}_{row.get('clean_text', '')[:50]}", axis=1
                )
                existing_keys = set(existing_sentiment['key'])
                news_to_process = news_df[~news_df['key'].isin(existing_keys)]
                
            self.logger.info(f"Найдено {len(news_to_process)} новых новостей для анализа")
            
            if news_to_process.empty:
                self.logger.info(f"Нет новых новостей для {ticker}, используем существующие данные")
                results['news_with_sentiment'] = existing_sentiment
                
                # Шаг 3: Обнаружение событий (используем существующие данные)
                events_file = os.path.join(output_dir, f"{ticker}_news_with_events.csv")
                if os.path.exists(events_file):
                    news_with_events = pd.read_csv(events_file)
                    daily_events = event_detector.create_event_time_series(news_with_events)
                    results['news_with_events'] = news_with_events
                    results['daily_events'] = daily_events
                else:
                    # Если файл событий не существует, создаем его
                    news_with_events = event_detector.detect_events(existing_sentiment)
                    news_with_events = event_detector.assess_event_impact(news_with_events)
                    news_with_events.to_csv(events_file, index=False)
                    daily_events = event_detector.create_event_time_series(news_with_events)
                    daily_events.to_csv(os.path.join(output_dir, f"{ticker}_daily_events.csv"), index=False)
                    results['news_with_events'] = news_with_events
                    results['daily_events'] = daily_events
                
                # Шаг 4: Используем существующие признаки или создаем новые
                daily_sentiment = sentiment_analyzer.create_daily_sentiment_series(existing_sentiment)
                sentiment_features = feature_extractor.create_time_series_features(daily_sentiment)
                results['daily_sentiment'] = daily_sentiment
                results['sentiment_features'] = sentiment_features
                
                # Переходим сразу к шагу 5 (интеграция с ценами)
                self.logger.info(f"Интеграция с ценами для {ticker}")
                ticker_dir = os.path.join(base_dir, 'data', 'processed_data', ticker)
                parquet_files = [f for f in os.listdir(ticker_dir) if f.endswith('.parquet') and ticker in f]
                
                if not parquet_files:
                    self.logger.warning(f"Ценовые данные для {ticker} не найдены")
                    return results
                    
                # Берем самый свежий parquet файл
                parquet_file = max(parquet_files, key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)))
                price_path = os.path.join(ticker_dir, parquet_file)
                
                try:
                    # Загрузка и агрегация цен
                    price_df = pd.read_parquet(price_path)
                    price_df['date'] = pd.to_datetime(price_df['date'])
                    
                    # Приведение к стандартным названиям столбцов
                    column_mapping = {'min': 'low', 'max': 'high'}
                    price_df = price_df.rename(columns=column_mapping)
                    
                    # Агрегация до дневных данных
                    price_df['day'] = price_df['date'].dt.date
                    daily_price = price_df.groupby('day').agg({
                        'open': 'first',
                        'close': 'last',
                        'high': 'max',
                        'low': 'min',
                        'volume': 'sum'
                    }).reset_index()
                    
                    daily_price = daily_price.rename(columns={'day': 'date'})
                    daily_price['date'] = pd.to_datetime(daily_price['date'])
                    
                    # Объединение с новостными признаками
                    combined_df = integrator.merge_news_with_price_data(
                        sentiment_features,
                        daily_price,
                        date_column='date'
                    )
                    
                    # Проверяем, что объединение успешно
                    if combined_df.empty:
                        self.logger.warning(f"Не удалось объединить новостные и ценовые данные для {ticker}")
                    else:
                        # Визуализация цен и настроений
                        plt.figure(figsize=(12, 6))
                        ax1 = plt.gca()
                        ax2 = ax1.twinx()
                        
                        # Получаем данные для графика
                        date_col = combined_df.index if isinstance(combined_df.index, pd.DatetimeIndex) else combined_df['date']
                        
                        # Построение графика
                        line1 = ax1.plot(date_col, combined_df['close'], 'b-', label='Цена')
                        line2 = ax2.plot(date_col, combined_df['avg_sentiment'], 'r-', label='Настроение')
                        
                        ax1.set_xlabel('Дата')
                        ax1.set_ylabel('Цена закрытия', color='b')
                        ax2.set_ylabel('Среднее настроение', color='r')
                        
                        # Объединение легенд
                        lines = line1 + line2
                        labels = ['Цена', 'Настроение']
                        ax1.legend(lines, labels, loc='upper left')
                        
                        plt.title(f'Сравнение цен и настроений для {ticker}')
                        plt.tight_layout()
                        
                        # Сохранение с явным указанием пути
                        chart_path = os.path.join(output_dir, f"{ticker}_price_vs_sentiment.png")
                        plt.savefig(chart_path)
                        self.logger.info(f"График сохранен в {chart_path}")
                        plt.close()
                    
                    # Создание признаков для ML
                    ml_df = combined_df.copy()
                    
                    # Целевая переменная
                    prediction_horizon = 5
                    ml_df['target_return'] = ml_df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
                    
                    # Новостные признаки
                    news_features = [col for col in ml_df.columns if col.startswith(('sentiment', 'news_count', 'event_', 'topic_'))]
                
                    # Инженерия признаков
                    feature_dict = {}
                    
                    # Лаги
                    for feature in news_features:
                        for lag in [1, 2, 3, 5, 10]:
                            feature_dict[f'{feature}_lag{lag}'] = ml_df[feature].shift(lag)
                    
                    # Скользящие средние
                    for feature in news_features:
                        for window in [3, 7, 14, 30]:
                            feature_dict[f'{feature}_ma{window}'] = ml_df[feature].rolling(window=window, min_periods=1).mean()
                        
                        # Волатильность
                        feature_dict[f'{feature}_std7'] = ml_df[feature].rolling(window=7, min_periods=1).std()
                        feature_dict[f'{feature}_std14'] = ml_df[feature].rolling(window=14, min_periods=1).std()
                        
                    # Комплексные признаки
                                        # Комплексные признаки
                    if 'avg_sentiment' in ml_df.columns:
                        feature_dict['sentiment_return_corr'] = ml_df['avg_sentiment'].rolling(window=14).corr(ml_df['close'].pct_change())
                    
                    if 'news_count' in ml_df.columns:
                        feature_dict['news_count_volatility'] = ml_df['news_count'] * ml_df['close'].pct_change().rolling(window=7).std()
                    
                    # Добавление всех признаков
                    ml_features = pd.concat([ml_df, pd.DataFrame(feature_dict)], axis=1).reset_index()
                    ml_features.to_csv(os.path.join(output_dir, f"{ticker}_ml_features.csv"))
                    
                    results['ml_features'] = ml_features
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке ценовых данных: {e}")
                    traceback.print_exc()
                    
                return results
        else:
            news_to_process = news_df
        
        # Шаг 2: Анализ настроений для новых новостей
        self.logger.info(f"Анализ настроений для {ticker}")
        news_with_sentiment = sentiment_analyzer.analyze_ticker_news(
            news_to_process,
            save_path=None  # Не сохраняем сразу, сначала объединим
        )
        
        # Объединяем с существующими данными, если они есть
        if os.path.exists(sentiment_file):
            combined_sentiment = pd.concat([existing_sentiment, news_with_sentiment], ignore_index=True)
            # Удаление дубликатов
            if 'id' in combined_sentiment.columns:
                combined_sentiment = combined_sentiment.drop_duplicates(subset='id')
            else:
                combined_sentiment = combined_sentiment.drop_duplicates(subset=['date', 'clean_text'])
        else:
            combined_sentiment = news_with_sentiment
        
        # Сохраняем обновленные данные
        combined_sentiment.to_csv(sentiment_file, index=False)
        self.logger.info(f"Сохранены обновленные данные с сентиментом для {ticker}: {len(combined_sentiment)} новостей")
        
        # Обновляем результаты
        results['news_with_sentiment'] = combined_sentiment
        
        # Создаем временной ряд из всех данных
        daily_sentiment = sentiment_analyzer.create_daily_sentiment_series(combined_sentiment)
        daily_sentiment.to_csv(os.path.join(output_dir, f"{ticker}_daily_sentiment.csv"), index=False)
        
        results['daily_sentiment'] = daily_sentiment
        
        # Визуализация настроений
        plt.figure(figsize=(10, 5))
        sns.countplot(x='sentiment_category', data=combined_sentiment)
        plt.title(f'Распределение настроений для {ticker}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_sentiment_distribution.png"))
        plt.close()
        
        # Шаг 3: Обнаружение событий
        self.logger.info(f"Обнаружение событий для {ticker}")
        news_with_events = event_detector.detect_events(combined_sentiment)
        news_with_events = event_detector.assess_event_impact(news_with_events)
        news_with_events.to_csv(os.path.join(output_dir, f"{ticker}_news_with_events.csv"), index=False)
        
        daily_events = event_detector.create_event_time_series(news_with_events)
        daily_events.to_csv(os.path.join(output_dir, f"{ticker}_daily_events.csv"), index=False)
        
        results['news_with_events'] = news_with_events
        results['daily_events'] = daily_events
        
        # Визуализация событий
        event_columns = [col for col in news_with_events.columns 
                        if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
        event_counts = {col.replace('event_', ''): news_with_events[col].sum() for col in event_columns}
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(event_counts.keys()), y=list(event_counts.values()))
        plt.title(f'Распределение типов событий для {ticker}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_event_distribution.png"))
        plt.close()
        
        # Шаг 4: Извлечение признаков
        self.logger.info(f"Извлечение признаков для {ticker}")
        try:
            news_with_topics, topic_dict = feature_extractor.create_topic_features(
            news_with_events,
            text_column='clean_text',
            n_topics=3
        )
        
            with open(os.path.join(output_dir, f"{ticker}_topics.txt"), 'w') as f:
                for topic, keywords in topic_dict.items():
                    f.write(f"{topic}: {keywords}\n")
                    
            results['topic_dict'] = topic_dict
            results['news_with_topics'] = news_with_topics
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении тем: {e}")
            results['topic_dict'] = {}
            results['news_with_topics'] = news_with_events
        
        # Извлечение временных признаков
        sentiment_features = feature_extractor.create_time_series_features(daily_sentiment)
        sentiment_features.to_csv(os.path.join(output_dir, f"{ticker}_sentiment_features.csv"), index=False)
        results['sentiment_features'] = sentiment_features
        
        # Визуализация временного ряда
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(sentiment_features['date'], sentiment_features['avg_sentiment'], label='Среднее настроение')
        plt.plot(sentiment_features['date'], sentiment_features['sentiment_ma_7d'], label='Скользящее среднее (7 дней)')
        plt.title(f'Динамика настроений для {ticker}')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(sentiment_features['date'], sentiment_features['news_count'], label='Количество новостей')
        plt.title(f'Объем новостей для {ticker}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_sentiment_dynamics.png"))
        plt.close()
        
        # Шаг 5: Интеграция с ценами
        self.logger.info(f"Интеграция с ценами для {ticker}")
        ticker_dir = os.path.join(base_dir, 'data', 'processed_data', ticker)
        parquet_files = [f for f in os.listdir(ticker_dir) if f.endswith('.parquet') and ticker in f]
        
        if not parquet_files:
            self.logger.warning(f"Ценовые данные для {ticker} не найдены")
            return results
            
        # Берем самый свежий parquet файл
        parquet_file = max(parquet_files, key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)))
        price_path = os.path.join(ticker_dir, parquet_file)
        
        try:
            # Загрузка и агрегация цен
            price_df = pd.read_parquet(price_path)
            price_df['date'] = pd.to_datetime(price_df['date'])
            
            # Приведение к стандартным названиям столбцов
            column_mapping = {'min': 'low', 'max': 'high'}
            price_df = price_df.rename(columns=column_mapping)
            
            # Агрегация до дневных данных
            price_df['day'] = price_df['date'].dt.date
            daily_price = price_df.groupby('day').agg({
                'open': 'first',
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum'
            }).reset_index()
            
            daily_price = daily_price.rename(columns={'day': 'date'})
            daily_price['date'] = pd.to_datetime(daily_price['date'])
            
            # Объединение с новостными признаками
            combined_df = integrator.merge_news_with_price_data(
                sentiment_features,
                daily_price,
                date_column='date'
            )
            
            # Проверяем, что объединение успешно
            if combined_df.empty:
                self.logger.warning(f"Не удалось объединить новостные и ценовые данные для {ticker}")
            else:
                # Визуализация цен и настроений
                plt.figure(figsize=(12, 6))
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                
                # Получаем данные для графика
                date_col = combined_df.index if isinstance(combined_df.index, pd.DatetimeIndex) else combined_df['date']
                
                # Построение графика
                line1 = ax1.plot(date_col, combined_df['close'], 'b-', label='Цена')
                line2 = ax2.plot(date_col, combined_df['avg_sentiment'], 'r-', label='Настроение')
                
                ax1.set_xlabel('Дата')
                ax1.set_ylabel('Цена закрытия', color='b')
                ax2.set_ylabel('Среднее настроение', color='r')
                
                # Объединение легенд
                lines = line1 + line2
                labels = ['Цена', 'Настроение']
                ax1.legend(lines, labels, loc='upper left')
                
                plt.title(f'Сравнение цен и настроений для {ticker}')
                plt.tight_layout()
                
                # Сохранение с явным указанием пути
                chart_path = os.path.join(output_dir, f"{ticker}_price_vs_sentiment.png")
                plt.savefig(chart_path)
                self.logger.info(f"График сохранен в {chart_path}")
                plt.close()
            
            # Создание признаков для ML
            ml_df = combined_df.copy()
            
            # Целевая переменная
            prediction_horizon = 5
            ml_df['target_return'] = ml_df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            
            # Новостные признаки
            news_features = [col for col in ml_df.columns if col.startswith(('sentiment', 'news_count', 'event_', 'topic_'))]
            
            # Инженерия признаков
            feature_dict = {}
            
            # Лаги
            for feature in news_features:
                for lag in [1, 2, 3, 5, 10]:
                    feature_dict[f'{feature}_lag{lag}'] = ml_df[feature].shift(lag)
            
            # Скользящие средние
            for feature in news_features:
                for window in [3, 7, 14, 30]:
                    feature_dict[f'{feature}_ma{window}'] = ml_df[feature].rolling(window=window, min_periods=1).mean()
                
                # Волатильность
                feature_dict[f'{feature}_std7'] = ml_df[feature].rolling(window=7, min_periods=1).std()
                feature_dict[f'{feature}_std14'] = ml_df[feature].rolling(window=14, min_periods=1).std()

            # Комплексные признаки
            if 'avg_sentiment' in ml_df.columns:
                feature_dict['sentiment_return_corr'] = ml_df['avg_sentiment'].rolling(window=14).corr(ml_df['close'].pct_change())
            
            if 'news_count' in ml_df.columns:
                feature_dict['news_count_volatility'] = ml_df['news_count'] * ml_df['close'].pct_change().rolling(window=7).std()
            
            # Добавление всех признаков
            ml_features = pd.concat([ml_df, pd.DataFrame(feature_dict)], axis=1).reset_index()
            ml_features.to_csv(os.path.join(output_dir, f"{ticker}_ml_features.csv"))
            
            results['ml_features'] = ml_features
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке ценовых данных: {e}")
            traceback.print_exc()
            
        return results
    
    def run_pipeline(
        self,
        base_dir: str = '/Users/aeshef/Documents/GitHub/kursach',
        tickers: List[str] = None,
        collect_telegram: bool = False,
        telegram_api_id: Optional[int] = None,
        telegram_api_hash: Optional[str] = None,
        telegram_channel: str = "cbrstocks",
        telegram_limit: int = 5000,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        use_cached_telegram: bool = True,
        cleanup_old_files: bool = False,  # Параметр для управления очисткой
        max_history_days: int = 30  # Сколько дней хранить старые файлы
    ):
        """Запуск полного пайплайна анализа новостей"""
        
        # Если включена очистка старых файлов
        if cleanup_old_files:
            self._cleanup_old_files(base_dir, tickers, max_history_days)

        self.logger.info(f"Запуск пайплайна анализа новостей для {len(tickers)} тикеров")
        self.logger.info(f"Диапазон дат: {start_date} - {end_date}")
        
        # Инициализация компонентов
        preprocessor = NewsPreprocessor(base_dir)
        sentiment_analyzer = SentimentAnalyzer(language='russian')
        feature_extractor = NewsFeatureExtractor()
        event_detector = EventDetector()
        integrator = NewsIntegration()
        
        # В методе run_pipeline добавить защиту от пустых данных
        # Шаг 1: Сбор данных из Telegram
        if collect_telegram:
            self.logger.info("=== СБОР ДАННЫХ ИЗ TELEGRAM ===")
            try:
                output_dir = os.path.join(base_dir, 'data', 'telegram_news')
                telegram_data = self._collect_telegram_data(
                    api_id=telegram_api_id,
                    api_hash=telegram_api_hash,
                    channel=telegram_channel,
                    limit=telegram_limit,
                    output_dir=output_dir,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Проверяем, получены ли данные
                if not telegram_data:
                    self.logger.warning("Данные из Telegram не получены, переходим к кэшированным данным")
                else:
                    # Сохраняем данные в директорию каждого тикера
                    for ticker, df in telegram_data.items():
                        print(f"СБОР ДАННЫХ ДЛЯ {ticker}")
                        ticker_dir = os.path.join(base_dir, 'data', 'processed_data', ticker)
                        os.makedirs(ticker_dir, exist_ok=True)
                        
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                        filename = f"{ticker}_news_{timestamp}.csv"
                        save_path = os.path.join(ticker_dir, filename)
                        df.to_csv(save_path, index=False, encoding='utf-8')
                        self.logger.info(f"Сохранены данные для {ticker}: {len(df)} новостей")
                        
                    self.telegram_data = telegram_data
            except Exception as e:
                self.logger.error(f"Ошибка при сборе данных из Telegram: {str(e)}")
                traceback.print_exc()
        
        # Шаг 2: Загрузка кэшированных данных
        if use_cached_telegram:
            self.logger.info("=== ЗАГРУЗКА КЭШИРОВАННЫХ ДАННЫХ ===")
            cached_data = self._load_telegram_data(
                data_dir=base_dir, 
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            self.telegram_data.update(cached_data)
        
        # Шаг 3: Обработка каждого тикера
        all_results = {}
        for ticker in tickers:
            self.logger.info(f"\n=== ОБРАБОТКА ТИКЕРА {ticker} ===")
            ticker_results = self._process_ticker(
                ticker=ticker,
                base_dir=base_dir,
                preprocessor=preprocessor,
                sentiment_analyzer=sentiment_analyzer,
                event_detector=event_detector,
                feature_extractor=feature_extractor,
                integrator=integrator
            )
            all_results[ticker] = ticker_results
            
        # Шаг 4: Создание сводного отчета
        self.logger.info("\n=== СОЗДАНИЕ СВОДНОГО ОТЧЕТА ===")
        summary_file = os.path.join(base_dir, 'data', 'news_analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== СВОДНЫЙ ОТЧЕТ ПО АНАЛИЗУ НОВОСТЕЙ ===\n\n")
            f.write(f"Дата анализа: {datetime.datetime.now()}\n")
            f.write(f"Период анализа: {start_date} - {end_date}\n\n")
            
            for ticker in tickers:
                if ticker not in all_results:
                    continue
                    
                results = all_results[ticker]
                f.write(f"Тикер: {ticker}\n")
                f.write("-" * 50 + "\n")
                
                if 'processed_news' in results:
                    f.write(f"Количество новостей: {len(results['processed_news'])}\n")
                    
                if 'news_with_sentiment' in results:
                    sentiment_df = results['news_with_sentiment']
                    sentiment_stats = sentiment_df['sentiment_category'].value_counts()
                    f.write("Распределение настроений:\n")
                    for category, count in sentiment_stats.items():
                        f.write(f"- {category}: {count} ({count/len(sentiment_df)*100:.1f}%)\n")
                
                if 'news_with_events' in results:
                    events_df = results['news_with_events']
                    event_columns = [col for col in events_df.columns 
                                    if col.startswith('event_') and col not in 
                                    ['event_impact', 'event_direction', 'has_event']]
                    
                    f.write("Статистика по типам событий:\n")
                    for col in event_columns:
                        event_count = events_df[col].sum()
                        event_name = col.replace('event_', '')
                        f.write(f"- {event_name}: {event_count} новостей\n")
                
                if 'ml_features' in results:
                    ml_df = results['ml_features']
                    f.write(f"Создан набор данных для ML моделей: {len(ml_df)} записей, {len(ml_df.columns)} признаков\n")
                
                f.write("\n")
                
        self.logger.info(f"Сводный отчет сохранен в {summary_file}")
        self.logger.info("\n=== АНАЛИЗ НОВОСТЕЙ ЗАВЕРШЕН ===")
        
        return all_results


def main():
    """Запуск пайплайна из командной строки"""
    parser = argparse.ArgumentParser(description='News Pipeline')
    
    # Основной параметр - файл конфигурации
    parser.add_argument('--config', type=str, 
                        help='Путь к файлу конфигурации JSON')
    
    # Дополнительные параметры для прямого запуска
    parser.add_argument('--base_dir', type=str, 
                        help='Базовая директория проекта')
    parser.add_argument('--tickers', type=str, nargs='+', 
                        help='Список тикеров для анализа')
    parser.add_argument('--start_date', type=str,
                        help='Начальная дата периода (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str,
                        help='Конечная дата периода (YYYY-MM-DD)')
    parser.add_argument('--collect_telegram', action='store_true',
                        help='Собирать данные из Telegram')
    parser.add_argument('--telegram_api_id', type=int,
                        help='Telegram API ID')
    parser.add_argument('--telegram_api_hash', type=str,
                        help='Telegram API Hash')
    parser.add_argument('--telegram_channel', type=str, default='cbrstocks',
                        help='Telegram канал для сбора новостей')
    parser.add_argument('--telegram_limit', type=int, default=5000,
                        help='Максимальное количество сообщений для сбора')
    parser.add_argument('--use_cached_telegram', action='store_true',
                        help='Использовать сохраненные данные Telegram')
    
    args = parser.parse_args()
    
    # Запуск через файл конфигурации
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            # Получаем параметры
            params = config.get('params', config)  # Совместимость с обоими форматами
            
            # Преобразование строк дат в объекты date
            if 'start_date' in params and isinstance(params['start_date'], str):
                params['start_date'] = datetime.datetime.strptime(params['start_date'], '%Y-%m-%d').date()
            if 'end_date' in params and isinstance(params['end_date'], str):
                params['end_date'] = datetime.datetime.strptime(params['end_date'], '%Y-%m-%d').date()
            
            # Получаем информацию о классе и методе (если есть)
            target = config.get('target')
            method = config.get('method')
            
            if target == 'NewsPipeline' and method == 'run_pipeline' or not target:
                # Стандартный запуск пайплайна
                pipeline = NewsPipeline()
                print(f"Запуск pipeline.run_pipeline с параметрами: {params}")
                pipeline.run_pipeline(**params)
            else:
                print(f"Неизвестная конфигурация: target={target}, method={method}")
                
        except Exception as e:
            print(f"Ошибка при чтении или применении конфигурации: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Прямой запуск с аргументами командной строки
    else:
        # Проверяем, что указаны минимально необходимые параметры
        if not args.base_dir:
            print("Необходимо указать базовую директорию (--base_dir)")
            return 1
        
        # Создаем словарь параметров из аргументов командной строки
        params = {
            'base_dir': args.base_dir
        }
        
        if args.tickers:
            params['tickers'] = args.tickers
        if args.start_date:
            params['start_date'] = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        if args.end_date:
            params['end_date'] = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()
        if args.collect_telegram:
            params['collect_telegram'] = args.collect_telegram
        if args.telegram_api_id:
            params['telegram_api_id'] = args.telegram_api_id
        if args.telegram_api_hash:
            params['telegram_api_hash'] = args.telegram_api_hash
        if args.telegram_channel:
            params['telegram_channel'] = args.telegram_channel
        if args.telegram_limit:
            params['telegram_limit'] = args.telegram_limit
        if args.use_cached_telegram:
            params['use_cached_telegram'] = args.use_cached_telegram
        
        # Запускаем пайплайн
        try:
            pipeline = NewsPipeline()
            pipeline.run_pipeline(**params)
        except Exception as e:
            print(f"Ошибка при запуске пайплайна: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())