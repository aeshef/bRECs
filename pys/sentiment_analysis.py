import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

class SentimentAnalyzer:
    """Класс для анализа настроений в новостях"""
    
    def __init__(self, use_vader=True, language='english'):
        """
        Инициализация анализатора настроений
        
        Args:
            use_vader (bool): Использовать ли VADER для анализа настроений
            language (str): Язык текста ('english' или 'russian')
        """
        self.use_vader = use_vader
        self.language = language
        
        if use_vader and language == 'english':
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.sid = SentimentIntensityAnalyzer()
        
        if language == 'russian':
            self.positive_words = set(['хорошо', 'отлично', 'прибыль', 'рост', 'позитивно', 'увеличение'])
            self.negative_words = set(['плохо', 'убыток', 'падение', 'негативно', 'снижение', 'ухудшение'])
    
    def analyze_text_vader(self, text):
        """
        Анализ настроений текста с помощью VADER
        
        Args:
            text (str): Текст для анализа
            
        Returns:
            dict: Результаты анализа настроений
        """
        if not isinstance(text, str) or not text:
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
        
        return self.sid.polarity_scores(text)
    
    def analyze_text_simple_ru(self, text):
        """
        Простой анализ настроений для русского текста на основе словаря
        
        Args:
            text (str): Текст для анализа
            
        Returns:
            dict: Результаты анализа настроений
        """
        if not isinstance(text, str) or not text:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
        
        words = text.lower().split()
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        pos_ratio = pos_count / total_words if total_words > 0 else 0
        neg_ratio = neg_count / total_words if total_words > 0 else 0
        
        compound = (pos_ratio - neg_ratio) * 2
        
        compound = max(-1, min(1, compound))
        
        neu_ratio = 1 - (pos_ratio + neg_ratio)
        
        return {
            'compound': compound,
            'neg': neg_ratio,
            'neu': neu_ratio,
            'pos': pos_ratio
        }
    
    def analyze_text(self, text):
        """
        Общий метод для анализа настроений текста
        
        Args:
            text (str): Текст для анализа
            
        Returns:
            dict: Результаты анализа настроений
        """
        if self.use_vader and self.language == 'english':
            return self.analyze_text_vader(text)
        elif self.language == 'russian':
            return self.analyze_text_simple_ru(text)
        else:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
    
    def analyze_news_dataframe(self, df, text_column='clean_text'):
        """
        Анализ настроений для всех новостей в DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame с новостями
            text_column (str): Название колонки с текстом для анализа
            
        Returns:
            pd.DataFrame: DataFrame с добавленными результатами анализа настроений
        """
        if df.empty or text_column not in df.columns:
            print(f"DataFrame пуст или не содержит колонку {text_column}")
            return df
        
        result_df = df.copy()

        sentiments = result_df[text_column].apply(self.analyze_text)
        
        result_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
        result_df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
        result_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
        result_df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
        
        result_df['sentiment_category'] = result_df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        return result_df
    
    def analyze_ticker_news(self, news_df, save_path=None):
        """
        Анализ настроений для новостей одного тикера
        
        Args:
            news_df (pd.DataFrame): DataFrame с новостями тикера
            save_path (str): Путь для сохранения результатов. Если None, результаты не сохраняются
            
        Returns:
            pd.DataFrame: DataFrame с результатами анализа настроений
        """
        analyzed_df = self.analyze_news_dataframe(news_df)
        
        if save_path is not None:
            analyzed_df.to_csv(save_path, index=False)
            print(f"Результаты анализа настроений сохранены в {save_path}")
        
        return analyzed_df
    
    def create_daily_sentiment_series(self, analyzed_df):
        """
        Создание временного ряда ежедневных настроений
        
        Args:
            analyzed_df (pd.DataFrame): DataFrame с результатами анализа настроений
            
        Returns:
            pd.DataFrame: DataFrame с ежедневными агрегированными настроениями
        """
        if 'date' not in analyzed_df.columns or 'sentiment_compound' not in analyzed_df.columns:
            print("DataFrame не содержит необходимые колонки (date, sentiment_compound)")
            return pd.DataFrame()
        
        analyzed_df['date'] = pd.to_datetime(analyzed_df['date'])
        
        daily_sentiment = analyzed_df.groupby(pd.Grouper(key='date', freq='D')).agg({
            'sentiment_compound': 'mean',
            'id': 'count'
        }).reset_index()
        
        daily_sentiment.rename(columns={'sentiment_compound': 'avg_sentiment', 'id': 'news_count'}, inplace=True)
        
        if not daily_sentiment.empty:
            date_range = pd.date_range(start=daily_sentiment['date'].min(), end=daily_sentiment['date'].max(), freq='D')
            daily_sentiment = daily_sentiment.set_index('date').reindex(date_range).fillna(0).reset_index()
            daily_sentiment.rename(columns={'index': 'date'}, inplace=True)
        
        return daily_sentiment
