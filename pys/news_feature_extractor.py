import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class NewsFeatureExtractor:
    """Класс для извлечения признаков из новостей"""
    
    def __init__(self):
        """Инициализация экстрактора признаков"""
        pass
    
    def extract_keyword_mentions(self, text, keywords):
        """
        Извлечение упоминаний ключевых слов из текста
        
        Args:
            text (str): Текст для анализа
            keywords (list): Список ключевых слов для поиска
            
        Returns:
            dict: Словарь {keyword: count} с количеством упоминаний каждого ключевого слова
        """
        if not isinstance(text, str) or not text:
            return {keyword: 0 for keyword in keywords}
        
        text_lower = text.lower()
        mentions = {}
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            mentions[keyword] = count
        
        return mentions
    
    def extract_topics_lda(self, texts, n_topics=5, n_top_words=10):
        """
        Извлечение тем из коллекции текстов с помощью LDA
        
        Args:
            texts (list): Список текстов для анализа
            n_topics (int): Количество тем для извлечения
            n_top_words (int): Количество ключевых слов для каждой темы
            
        Returns:
            tuple: (lda_model, vectorizer, document_topics, topic_keywords)
        """
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        
        dtm = vectorizer.fit_transform(texts)
        
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )
        
        document_topics = lda_model.fit_transform(dtm)
        
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_keywords.append(top_words)
        
        return lda_model, vectorizer, document_topics, topic_keywords
    
    def create_topic_features(self, df, text_column='clean_text', n_topics=5):
        """
        Создание признаков на основе тем для всех новостей в DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame с новостями
            text_column (str): Название колонки с текстом для анализа
            n_topics (int): Количество тем для извлечения
            
        Returns:
            pd.DataFrame: DataFrame с добавленными признаками тем
        """
        if df.empty or text_column not in df.columns:
            print(f"DataFrame пуст или не содержит колонку {text_column}")
            return df
        
        texts = df[text_column].fillna('').tolist()
        
        lda_model, vectorizer, document_topics, topic_keywords = self.extract_topics_lda(
            texts, n_topics=n_topics
        )
        
        result_df = df.copy()
        
        for i in range(n_topics):
            result_df[f'topic_{i}'] = document_topics[:, i]
        
        result_df['dominant_topic'] = np.argmax(document_topics, axis=1)
        
        topic_dict = {f'topic_{i}': ', '.join(words) for i, words in enumerate(topic_keywords)}
        
        return result_df, topic_dict
    
    def create_time_series_features(self, daily_sentiment_df, window_sizes=[3, 7, 14]):
        """
        Создание признаков временных рядов из ежедневных данных о настроениях
        
        Args:
            daily_sentiment_df (pd.DataFrame): DataFrame с ежедневными настроениями
            window_sizes (list): Список размеров окон для скользящих статистик
            
        Returns:
            pd.DataFrame: DataFrame с добавленными признаками временных рядов
        """
        if daily_sentiment_df.empty:
            print("DataFrame с ежедневными настроениями пуст")
            return daily_sentiment_df
        
        result_df = daily_sentiment_df.copy()
        
        for window in window_sizes:
            result_df[f'sentiment_ma_{window}d'] = result_df['avg_sentiment'].rolling(window=window, min_periods=1).mean()
            result_df[f'news_count_ma_{window}d'] = result_df['news_count'].rolling(window=window, min_periods=1).mean()
        
        for window in window_sizes:
            result_df[f'sentiment_std_{window}d'] = result_df['avg_sentiment'].rolling(window=window, min_periods=1).std()
   
        result_df['sentiment_change_1d'] = result_df['avg_sentiment'].diff()
        result_df['news_count_change_1d'] = result_df['news_count'].diff()
        
        for window in window_sizes:
            mean = result_df['avg_sentiment'].rolling(window=window, min_periods=1).mean()
            std = result_df['avg_sentiment'].rolling(window=window, min_periods=1).std().replace(0, 1)
            result_df[f'sentiment_zscore_{window}d'] = (result_df['avg_sentiment'] - mean) / std
        
        result_df = result_df.fillna(0)
        
        return result_df
