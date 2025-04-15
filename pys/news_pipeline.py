import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import importlib
from typing import List

import news_preprocessor
import sentiment_analysis
import news_feature_extractor
import event_detector
import news_visualizer
import news_integration

from news_preprocessor import NewsPreprocessor
from sentiment_analysis import SentimentAnalyzer
from news_feature_extractor import NewsFeatureExtractor
from event_detector import EventDetector
from news_visualizer import NewsVisualizer
from news_integration import NewsIntegration

import warnings
warnings.filterwarnings('ignore')

class NewsPipeline:
    def __init__(self):
        pass

    def run_pipeline(self, 
                    base_dir: str = '/Users/aeshef/Documents/GitHub/kursach',
                    tickers: List[str] = [
                            'SBER',  # Сбербанк
                            'GAZP',  # Газпром
                            'LKOH',  # Лукойл
                            'GMKN',  # ГМК "Норильский никель"
                            'ROSN',  # Роснефть
                            'TATN',  # Татнефть
                            'MTSS',  # МТС
                            'ALRS',  # АК Алроса
                            'SNGS',  # Сургутнефтегаз
                            'VTBR',  # ВТБ
                            'NVTK',  # Новатэк
                            'POLY',  # Полиметалл
                            'MVID',  # М.Видео
                            'PHOR',  # ФосАгро
                            'SIBN',  # Сибнефть
                            'AFKS',  # АФК Система
                            'MAGN',  # Магнитогорский металлургический комбинат
                            'RUAL']):

        preprocessor = NewsPreprocessor(base_dir)
        sentiment_analyzer = SentimentAnalyzer(language='russian')
        feature_extractor = NewsFeatureExtractor()
        event_detector = EventDetector()
        visualizer = NewsVisualizer()

        print("=== ПРЕДОБРАБОТКА НОВОСТЕЙ ===")
        processed_news = {}

        for ticker in tickers:
            print(f"\nПредобработка новостей для тикера {ticker}...")
            
            output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            news_df = preprocessor.process_ticker_news(ticker, save=True)
            
            if not news_df.empty:
                processed_news[ticker] = news_df
                print(f"Обработано {len(news_df)} новостей для {ticker}")
                
                if len(news_df) > 0:
                    print("Пример обработанной новости:")
                    first_news = news_df.iloc[0]
                    print(f"Дата: {first_news.get('date', 'Нет даты')}")
                    print(f"Текст до: {first_news.get('text', '')[:100]}...")
                    print(f"Текст после: {first_news.get('clean_text', '')}[:100]...")
            else:
                print(f"Нет новостей для тикера {ticker} или произошла ошибка при их обработке")


        print("\n=== АНАЛИЗ НАСТРОЕНИЙ ===")
        sentiment_results = {}

        for ticker, news_df in processed_news.items():
            print(ticker)
            print(f"\nАнализ настроений для тикера {ticker}...")
            
            output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
            
            news_with_sentiment = sentiment_analyzer.analyze_ticker_news(
                news_df, 
                save_path=os.path.join(output_dir, f"{ticker}_news_with_sentiment.csv")
            )
            
            daily_sentiment = sentiment_analyzer.create_daily_sentiment_series(news_with_sentiment)
            daily_sentiment.to_csv(os.path.join(output_dir, f"{ticker}_daily_sentiment.csv"), index=False)
            
            sentiment_results[ticker] = {
                'news_with_sentiment': news_with_sentiment,
                'daily_sentiment': daily_sentiment
            }
            
            print(f"Обработано настроение для {len(news_with_sentiment)} новостей")
            
            sentiment_counts = news_with_sentiment['sentiment_category'].value_counts()
            print("Распределение настроений:")
            print(sentiment_counts)
            
            plt.figure(figsize=(10, 5))
            sns.countplot(x='sentiment_category', data=news_with_sentiment)
            plt.title(f'Распределение настроений для {ticker}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ticker}_sentiment_distribution.png"))
            plt.close()


        print("\n=== ОБНАРУЖЕНИЕ СОБЫТИЙ ===")
        events_results = {}

        for ticker, results in sentiment_results.items():
            print(f"\nОбнаружение событий для тикера {ticker}...")
            
            news_with_sentiment = results['news_with_sentiment']
            
            output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
            
            news_with_events = event_detector.detect_events(news_with_sentiment)
            news_with_events = event_detector.assess_event_impact(news_with_events)
            news_with_events.to_csv(os.path.join(output_dir, f"{ticker}_news_with_events.csv"), index=False)
            
            daily_events = event_detector.create_event_time_series(news_with_events)
            daily_events.to_csv(os.path.join(output_dir, f"{ticker}_daily_events.csv"), index=False)
            
            events_results[ticker] = {
                'news_with_events': news_with_events,
                'daily_events': daily_events
            }
            
            event_columns = [col for col in news_with_events.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
            
            print("Статистика по типам событий:")
            for col in event_columns:
                event_count = news_with_events[col].sum()
                event_name = col.replace('event_', '')
                print(f"- {event_name}: {event_count} новостей")
            
            event_counts = {col.replace('event_', ''): news_with_events[col].sum() for col in event_columns}
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(event_counts.keys()), y=list(event_counts.values()))
            plt.title(f'Распределение типов событий для {ticker}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ticker}_event_distribution.png"))
            plt.close()


        print("\n=== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ДЛЯ МОДЕЛЕЙ ===")
        features_results = {}

        for ticker, events in events_results.items():
            print(f"\nСоздание признаков для тикера {ticker}...")
            
            news_with_events = events['news_with_events']
            daily_sentiment = sentiment_results[ticker]['daily_sentiment']
            
            output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
            
            try:
                news_with_topics, topic_dict = feature_extractor.create_topic_features(
                    news_with_events, 
                    text_column='clean_text',
                    n_topics=3
                )
                
                with open(os.path.join(output_dir, f"{ticker}_topics.txt"), 'w') as f:
                    for topic, keywords in topic_dict.items():
                        f.write(f"{topic}: {keywords}\n")
                        
                print(f"Выделено {len(topic_dict)} тем из новостей")
                print("Ключевые слова для первой темы:", topic_dict.get('topic_0', 'Нет данных'))
                
            except Exception as e:
                print(f"Ошибка при извлечении тем: {e}")
                news_with_topics = news_with_events
                topic_dict = {}
            
            sentiment_features = feature_extractor.create_time_series_features(daily_sentiment)
            sentiment_features.to_csv(os.path.join(output_dir, f"{ticker}_sentiment_features.csv"), index=False)
            
            features_results[ticker] = {
                'news_with_topics': news_with_topics,
                'topic_dict': topic_dict,
                'sentiment_features': sentiment_features
            }
            
            print(f"Создано {len(sentiment_features.columns)} признаков временных рядов")
            
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


        print("\n=== ИНТЕГРАЦИЯ С ЦЕНОВЫМИ ДАННЫМИ ===")
        ml_features_results = {}

        integrator = NewsIntegration()

        for ticker, features in features_results.items():
            print(f"\nИнтеграция новостей с ценами для тикера {ticker}...")
            
            sentiment_features = features['sentiment_features']
            
            output_dir = os.path.join(base_dir, 'data', 'processed_data', ticker, 'news_analysis')
            
            ticker_dir = os.path.join(base_dir, 'data', 'processed_data', ticker)
            parquet_files = [f for f in os.listdir(ticker_dir) if f.endswith('.parquet') and ticker in f]
            
            if not parquet_files:
                print(f"Ценовые данные в формате parquet для {ticker} не найдены")
                continue
            
            parquet_file = sorted(parquet_files, key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)), reverse=True)[0]
            price_path = os.path.join(ticker_dir, parquet_file)
            
            print(f"Найден файл с ценами: {parquet_file}")
            
            try:
                import pandas as pd
                

                price_df = pd.read_parquet(price_path)
                
                print(f"Структура данных из parquet файла: {price_df.columns.tolist()}")
                
                price_df['date'] = pd.to_datetime(price_df['date'])
                
                column_mapping = {'min': 'low', 'max': 'high'}
                price_df = price_df.rename(columns=column_mapping)
                
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
                
                print("Пример агрегированных дневных данных:")
                print(daily_price.head(3))
                
                combined_df = integrator.merge_news_with_price_data(
                    sentiment_features,
                    daily_price,
                    date_column='date'
                )
                
                print(f"Структура объединенных данных: {combined_df.shape}, колонки: {combined_df.columns.tolist()[:5]}...")
                
                try:
                    ml_df = combined_df.copy()
                    
                    prediction_horizon = 5
                    ml_df['target_return'] = ml_df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
                    
                    news_features = [col for col in ml_df.columns if col.startswith(('sentiment', 'news_count', 'event_', 'topic_'))]
                    
                    new_columns = {}
                    
                    for feature in news_features:
                        for lag in [1, 2, 3, 5, 10]:
                            new_columns[f'{feature}_lag{lag}'] = ml_df[feature].shift(lag)
                    
                    for feature in news_features:
                        for window in [3, 7, 14, 30]:
                            new_columns[f'{feature}_ma{window}'] = ml_df[feature].rolling(window=window, min_periods=1).mean()
                        
                        new_columns[f'{feature}_std7'] = ml_df[feature].rolling(window=7, min_periods=1).std()
                        new_columns[f'{feature}_std14'] = ml_df[feature].rolling(window=14, min_periods=1).std()

                    if 'avg_sentiment' in ml_df.columns:
                        new_columns['sentiment_return_corr'] = ml_df['avg_sentiment'].rolling(window=14).corr(ml_df['close'].pct_change())
                    
                    if 'news_count' in ml_df.columns:
                        new_columns['news_count_volatility'] = ml_df['news_count'] * ml_df['close'].pct_change().rolling(window=7).std()
                    
                    ml_df = pd.concat([ml_df, pd.DataFrame(new_columns)], axis=1)
                    
                    ml_df = ml_df.reset_index()
                    ml_features = ml_df
                    
                    print(f"Структура данных ML: {ml_features.shape}")
                    if 'date' in ml_features.columns:
                        print(f"Временной период: с {ml_features['date'].min()} по {ml_features['date'].max()}")
                    else:
                        print(f"Колонки в ml_features: {ml_features.columns.tolist()[:5]}...")
                    
                    ml_features.to_csv(os.path.join(output_dir, f"{ticker}_ml_features.csv"))
                    ml_features_results[ticker] = ml_features
                    
                    try:
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
                        
                        print(f"Создан график сравнения цен и настроений")
                    except Exception as e:
                        print(f"Ошибка при создании визуализации: {e}")
                        import traceback
                        traceback.print_exc()
                        
                except Exception as e:
                    print(f"Ошибка при создании признаков ML: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"Ошибка при интеграции данных: {e}")
                import traceback
                traceback.print_exc()



        print("\n=== СОЗДАНИЕ СВОДНОГО ОТЧЕТА ===")

        summary_file = os.path.join(base_dir, 'data', 'news_analysis_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== СВОДНЫЙ ОТЧЕТ ПО АНАЛИЗУ НОВОСТЕЙ ===\n\n")
            f.write(f"Дата анализа: {datetime.datetime.now()}\n\n")
            
            for ticker in tickers:
                f.write(f"Тикер: {ticker}\n")
                f.write("-" * 50 + "\n")
                
                if ticker in processed_news:
                    news_count = len(processed_news[ticker])
                    f.write(f"Количество новостей: {news_count}\n")
                else:
                    f.write("Новости не обработаны\n")
                
                if ticker in sentiment_results:
                    sentiment_df = sentiment_results[ticker]['news_with_sentiment']
                    sentiment_stats = sentiment_df['sentiment_category'].value_counts()
                    f.write("Распределение настроений:\n")
                    for category, count in sentiment_stats.items():
                        f.write(f"- {category}: {count} ({count/len(sentiment_df)*100:.1f}%)\n")
                
                if ticker in events_results:
                    events_df = events_results[ticker]['news_with_events']
                    event_columns = [col for col in events_df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
                    
                    f.write("Статистика по типам событий:\n")
                    for col in event_columns:
                        event_count = events_df[col].sum()
                        event_name = col.replace('event_', '')
                        f.write(f"- {event_name}: {event_count} новостей\n")
                
                if ticker in ml_features_results:
                    ml_df = ml_features_results[ticker]
                    f.write(f"Создан набор данных для ML моделей: {len(ml_df)} записей, {len(ml_df.columns)} признаков\n")
                
                f.write("\n")

        print(f"Сводный отчет сохранен в {summary_file}")

        print("\n=== АНАЛИЗ НОВОСТЕЙ ЗАВЕРШЕН ===")