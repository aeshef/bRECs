import pandas as pd
import numpy as np
import re

class EventDetector:
    """Класс для обнаружения событий в новостях"""
    
    def __init__(self):
        """Инициализация детектора событий"""
        self.event_keywords = {
            'earnings': ['отчет', 'прибыль', 'убыток', 'квартал', 'финансов', 'ebitda', 'выручка', 'ожидания', 'консенсус'],
            'dividends': ['дивиденд', 'выплата', 'дивидендная политика', 'рекомендация'],
            'mergers_acquisitions': ['слияние', 'поглощение', 'приобретение', 'm&a', 'сделка', 'покупка компании'],
            'regulatory': ['регулятор', 'цб', 'фас', 'минфин', 'центробанк', 'регуляторн', 'лицензия', 'ограничения', 'санкции'],
            'management': ['ceo', 'cfo', 'директор', 'руководитель', 'руководство', 'назначение', 'отставка', 'совет директоров'],
            'products': ['новый продукт', 'запуск', 'релиз', 'разработка', 'инновация', 'патент', 'технология'],
            'macroeconomic': ['инфляция', 'ввп', 'экономический рост', 'безработица', 'ставка', 'кризис', 'санкции', 'экономика']
        }
    
    def detect_events(self, df, text_column='clean_text'):
        """
        Обнаружение событий в новостях
        
        Args:
            df (pd.DataFrame): DataFrame с новостями
            text_column (str): Название колонки с текстом для анализа
            
        Returns:
            pd.DataFrame: DataFrame с добавленными колонками событий
        """
        if df.empty or text_column not in df.columns:
            print(f"DataFrame пуст или не содержит колонку {text_column}")
            return df
        
        result_df = df.copy()
        
        for event_type, keywords in self.event_keywords.items():
            def has_keywords(text):
                if not isinstance(text, str) or not text:
                    return False
                text_lower = text.lower()
                return any(keyword.lower() in text_lower for keyword in keywords)
            
            result_df[f'event_{event_type}'] = result_df[text_column].apply(has_keywords)
        
        event_columns = [col for col in result_df.columns if col.startswith('event_')]
        result_df['has_event'] = result_df[event_columns].any(axis=1)
        
        return result_df
    
    def assess_event_impact(self, df, sentiment_column='sentiment_compound'):
        """
        Оценка влияния событий на основе настроения и типа события
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями
            sentiment_column (str): Название колонки с оценкой настроения
            
        Returns:
            pd.DataFrame: DataFrame с добавленной оценкой влияния событий
        """
        if df.empty or sentiment_column not in df.columns:
            print(f"DataFrame пуст или не содержит колонку {sentiment_column}")
            return df
        
        result_df = df.copy()
        
        def calculate_impact(row):
            impact_score = 0
            
            event_types = [col for col in row.index if col.startswith('event_') and col != 'has_event']
            
            for event_type in event_types:
                if row[event_type]:
                    if 'earnings' in event_type:
                        impact_score += 2
                    elif 'dividends' in event_type:
                        impact_score += 2
                    elif 'mergers_acquisitions' in event_type:
                        impact_score += 3
                    elif 'regulatory' in event_type:
                        impact_score += 2
                    elif 'management' in event_type:
                        impact_score += 1
                    elif 'macroeconomic' in event_type:
                        impact_score += 1
            sentiment_strength = abs(row[sentiment_column])
            if sentiment_strength > 0.5:
                impact_score += 2
            elif sentiment_strength > 0.3:
                impact_score += 1
            
            if impact_score >= 8:
                return 5
            elif impact_score >= 6:
                return 4
            elif impact_score >= 4:
                return 3
            elif impact_score >= 2:
                return 2
            else:
                return 1
        
        result_df['event_impact'] = result_df.apply(calculate_impact, axis=1)
        result_df['event_direction'] = result_df[sentiment_column].apply(
            lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral')
        )
        
        return result_df
    
    def create_event_time_series(self, df, date_column='date'):
        """
        Создание временного ряда событий
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями
            date_column (str): Название колонки с датой
            
        Returns:
            pd.DataFrame: DataFrame с ежедневной агрегацией событий
        """
        if df.empty or date_column not in df.columns or 'has_event' not in df.columns:
            print("DataFrame пуст или не содержит необходимые колонки")
            return pd.DataFrame()
        
        df[date_column] = pd.to_datetime(df[date_column])
        
        event_columns = [col for col in df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction']]
        
        daily_events = df.groupby(pd.Grouper(key=date_column, freq='D')).agg({
            **{col: 'sum' for col in event_columns},
            'has_event': 'sum',
            'event_impact': 'mean'
        }).reset_index()
        
        if not daily_events.empty:
            date_range = pd.date_range(start=daily_events[date_column].min(), end=daily_events[date_column].max(), freq='D')
            daily_events = daily_events.set_index(date_column).reindex(date_range).fillna(0).reset_index()
            daily_events.rename(columns={'index': date_column}, inplace=True)
        
        return daily_events
