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

from news_preprocessor import NewsPreprocessor
from sentiment_analysis import SentimentAnalyzer
from news_feature_extractor import NewsFeatureExtractor
from event_detector import EventDetector
from news_visualizer import NewsVisualizer
from news_integration import NewsIntegration

import warnings
warnings.filterwarnings('ignore')

class NewsPipeline:
    """Единый пайплайн для сбора и анализа новостей"""
    
    def __init__(self):
        self.telegram_data = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Настройка логгера"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger('NewsPipeline')

    def _collect_telegram_data(
        self,
        api_id: Optional[int],
        api_hash: Optional[str],
        channel: str,
        limit: int,
        output_dir: str,
        tickers: List[str],
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None
    ) -> Dict[str, pd.DataFrame]:
        """Сбор данных из Telegram"""
        if not api_id or not api_hash:
            self.logger.warning("API данные не предоставлены. Пропускаем сбор данных Telegram.")
            return {}
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Компании и ключевые слова для поиска
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
        
        # Фильтруем только тикеры, которые нам нужны
        company_info = {ticker: info for ticker, info in COMPANY_INFO.items() 
                        if ticker in tickers}
        
        results = {}
        try:
            with TelegramClient('telegram_session', api_id, api_hash) as client:
                print("ХУИИИИ")
                print("Подключились к Telegram, получаем сущность канала...", flush=True)
                entity = client.get_entity(channel)
                print(f"Сущность: {entity}", flush=True)

                self.logger.info(f"Подключено к Telegram API")
                entity = client.get_entity(channel)
                
                # Собираем сообщения
                messages = []
                for message in client.iter_messages(entity, limit=limit):
                    msg_date = message.date.date()
                    
                    if start_date and msg_date < start_date:
                        continue
                    if end_date and msg_date > end_date:
                        continue
                    
                    messages.append(message)
                    
                    # Уменьшаем частоту API запросов
                    if len(messages) % 100 == 0:
                        self.logger.info(f"Получено {len(messages)} сообщений...")
                        print(f"Получено {len(messages)} сообщений...", flush=True)
                        time.sleep(1)  # Пауза для избежания ограничений API
                
                self.logger.info(f"Всего получено {len(messages)} сообщений")
                
                # Обработка сообщений
                message_data = []
                for msg in messages:
                    if not msg.text:
                        continue
                    
                    # Поиск тикеров в тегах
                    ticker_pattern = r'#([A-Z0-9]{4,6})'
                    tickers_from_tags = [t for t in re.findall(ticker_pattern, msg.text) 
                                        if t in company_info]
                    
                    # Поиск тикеров по ключевым словам
                    tickers_from_keywords = []
                    if not tickers_from_tags:
                        text_lower = msg.text.lower()
                        for ticker, info in company_info.items():
                            # Проверка названия компании
                            if info['name'].lower() in text_lower:
                                tickers_from_keywords.append(ticker)
                                continue
                            
                            # Проверка ключевых слов
                            for keyword in info['keywords']:
                                if keyword in text_lower:
                                    tickers_from_keywords.append(ticker)
                                    break
                    
                    all_tickers = list(set(tickers_from_tags + tickers_from_keywords))
                    if not all_tickers:
                        continue  # Пропускаем новости без тикеров
                        
                    message_data.append({
                        'id': msg.id,
                        'date': msg.date,
                        'text': msg.text,
                        'tickers': all_tickers,
                        'tickers_from_tags': tickers_from_tags,
                        'tickers_from_keywords': tickers_from_keywords,
                        'has_media': msg.media is not None
                    })
                
                # Создаем DataFrame и сохраняем общий файл
                df = pd.DataFrame(message_data)
                if not df.empty:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    all_messages_file = os.path.join(output_dir, f"telegram_headlines_{timestamp}.csv")
                    df.to_csv(all_messages_file, index=False, encoding='utf-8')
                    self.logger.info(f"Сохранено {len(df)} сообщений в файл {all_messages_file}")
                
                # Разделяем по тикерам
                for ticker in tickers:
                    if ticker not in company_info:
                        continue
                        
                    ticker_messages = df[df['tickers'].apply(
                        lambda x: isinstance(x, list) and ticker in x
                    )].copy()
                    
                    if ticker_messages.empty:
                        continue
                    
                    # Добавляем информацию
                    ticker_messages['ticker'] = ticker
                    ticker_messages['company_name'] = company_info[ticker]['name']
                    ticker_messages['industry'] = company_info[ticker]['industry']
                    ticker_messages['news_type'] = ticker_messages.apply(
                        lambda row: 'company_specific' if ticker in row.get('tickers_from_tags', []) 
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
                    for col in ['tickers', 'tickers_from_tags', 'tickers_from_keywords']:
                        if col in df.columns:
                            df[col] = df[col].apply(
                                lambda x: eval(x) if isinstance(x, str) and pd.notna(x) else x
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
        
        # Шаг 1: Предобработка
        self.logger.info(f"Предобработка новостей для {ticker}")
        news_df = preprocessor.process_ticker_news(ticker, save=True)
        
        if news_df.empty:
            self.logger.warning(f"Нет новостей для {ticker}")
            return results
            
        results['processed_news'] = news_df
        
        # Шаг 2: Анализ настроений
        self.logger.info(f"Анализ настроений для {ticker}")
        news_with_sentiment = sentiment_analyzer.analyze_ticker_news(
            news_df,
            save_path=os.path.join(output_dir, f"{ticker}_news_with_sentiment.csv")
        )
        daily_sentiment = sentiment_analyzer.create_daily_sentiment_series(news_with_sentiment)
        daily_sentiment.to_csv(os.path.join(output_dir, f"{ticker}_daily_sentiment.csv"), index=False)
        
        results['news_with_sentiment'] = news_with_sentiment
        results['daily_sentiment'] = daily_sentiment
        
        # Визуализация настроений
        plt.figure(figsize=(10, 5))
        sns.countplot(x='sentiment_category', data=news_with_sentiment)
        plt.title(f'Распределение настроений для {ticker}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker}_sentiment_distribution.png"))
        plt.close()
        
        # Шаг 3: Обнаружение событий
        self.logger.info(f"Обнаружение событий для {ticker}")
        news_with_events = event_detector.detect_events(news_with_sentiment)
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
            
            # Визуализация цен и настроений
            date_col = 'date' if 'date' in ml_features.columns else 'index'
            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            ax1.plot(ml_features[date_col], ml_features['close'], 'b-', label='Цена')
            ax2.plot(ml_features[date_col], ml_features['avg_sentiment'], 'r-', label='Настроение')
            
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Цена закрытия', color='b')
            ax2.set_ylabel('Среднее настроение', color='r')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f'Сравнение цен и настроений для {ticker}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ticker}_price_vs_sentiment.png"))
            plt.close()
            
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
        use_cached_telegram: bool = True
    ):
        """Запуск полного пайплайна анализа новостей"""

        self.logger.info(f"Запуск пайплайна анализа новостей для {len(tickers)} тикеров")
        self.logger.info(f"Диапазон дат: {start_date} - {end_date}")
        
        # Инициализация компонентов
        preprocessor = NewsPreprocessor(base_dir)
        sentiment_analyzer = SentimentAnalyzer(language='russian')
        feature_extractor = NewsFeatureExtractor()
        event_detector = EventDetector()
        integrator = NewsIntegration()
        
        # Шаг 1: Сбор данных из Telegram
        if collect_telegram:
            self.logger.info("=== СБОР ДАННЫХ ИЗ TELEGRAM ===")
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