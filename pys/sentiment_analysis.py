import pandas as pd
import numpy as np
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
import datetime
from collections import Counter

class SentimentAnalyzer:
    """Класс для анализа настроений в новостях"""
    
    def __init__(self, use_vader=True, language='english'):
        self.use_vader = use_vader
        self.language = language
        
        self.cbr_meetings = [
            "2024-02-16", "2024-03-22", "2024-04-26", 
            "2024-06-07", "2024-07-26", "2024-09-13", 
            "2024-10-25", "2024-12-20",
            "2025-02-14", "2025-03-21", "2025-04-25", 
            "2025-06-06", "2025-07-25",
        ]
        
        if use_vader and language == 'english':
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.sid = SentimentIntensityAnalyzer()
        
        if language == 'russian':
            self.positive_words = {
                'прибыль': 0.8, 'выручка': 0.7, 'доход': 0.7, 'рост': 0.7, 'увеличение': 0.6, 
                'подъем': 0.6, 'укрепление': 0.7, 'устойчивость': 0.6, 'стабильность': 0.5,
                'дивиденды': 0.8, 'капитализация': 0.7, 'рентабельность': 0.7, 'эффективность': 0.6,
                
                'приобретение': 0.5, 'расширение': 0.5, 'партнерство': 0.6, 'сделка': 0.4,
                'модернизация': 0.6, 'инновация': 0.6, 'разработка': 0.4, 'импортозамещение': 0.7,
                'экспорт': 0.6, 'контракт': 0.5, 'соглашение': 0.4, 'лицензия': 0.5, 
                
                'позитивно': 0.6, 'успешно': 0.5, 'эффективно': 0.5, 'положительно': 0.5,
                'перспективно': 0.4, 'оптимистично': 0.5, 'выгодно': 0.6, 'благоприятно': 0.5,
                'превосходит': 0.7, 'улучшение': 0.6, 'превышение': 0.6, 'восстановление': 0.5,
                
                'импортозамещение': 0.7, 'господдержка': 0.7, 'субсидия': 0.6, 'госзаказ': 0.6,
                'нацпроект': 0.5, 'приоритетный': 0.4, 'стратегический': 0.5
            }
            
            self.negative_words = {
                'убыток': 0.8, 'потеря': 0.7, 'падение': 0.7, 'снижение': 0.7, 'сокращение': 0.6,
                'обвал': 0.9, 'кризис': 0.8, 'дефолт': 0.9, 'банкротство': 0.9, 'задолженность': 0.7,
                'штраф': 0.7, 'санкции': 0.8, 'инфляция': 0.6, 'девальвация': 0.7, 'дефицит': 0.6,
                
                'реструктуризация': 0.5, 'увольнение': 0.7, 'сокращение штата': 0.7, 'отзыв лицензии': 0.8,
                'приостановка': 0.6, 'отказ': 0.6, 'закрытие': 0.7, 'приостановление': 0.6,
                'расследование': 0.6, 'проверка': 0.4, 'нарушение': 0.6, 'иск': 0.6, 'суд': 0.5,
                
                'негативно': 0.6, 'неудовлетворительно': 0.7, 'проблемно': 0.5, 'трудность': 0.5,
                'риск': 0.5, 'неопределенность': 0.5, 'угроза': 0.6, 'ухудшение': 0.7,
                'недостаточно': 0.4, 'замедление': 0.5, 'противоречивый': 0.4,
                
                'санкции': 0.8, 'антироссийский': 0.7, 'эмбарго': 0.7, 'блокировка': 0.7,
                'заморозка активов': 0.8, 'внешнеполитическое давление': 0.7, 'ограничения': 0.6
            }
            
            self.context_dict = {
                'ключевая ставка': {
                    'снижение': 0.7,
                    'повышение': -0.7,
                    'сохранение': -0.2,
                },
                'нефть': {
                    'рост цен': 0.7,
                    'падение цен': -0.7,
                    'стабилизация': 0.3,
                },
                'санкции': {
                    'новые': -0.8,
                    'ужесточение': -0.8,
                    'снятие': 0.9,
                    'ослабление': 0.7,
                    'обход': 0.5,
                    'адаптация': 0.4,
                },
                'война': {
                    'эскалация': -0.9,
                    'деэскалация': 0.8,
                    'переговоры': 0.6,
                },
                'вооруженный конфликт': {
                    'обострение': -0.8,
                    'прекращение': 0.8,
                },
                'торговая война': {
                    'начало': -0.7,
                    'окончание': 0.7,
                },
                'тарифы': {
                    'повышение': -0.6,
                    'снижение': 0.6,
                },
                'регулирование': {
                    'ужесточение': -0.5,
                    'упрощение': 0.6,
                },
                'налоги': {
                    'повышение': -0.7,
                    'снижение': 0.8,
                    'льготы': 0.7,
                },
                'ипотека': {
                    'снижение ставок': 0.7,
                    'повышение ставок': -0.7
                }
            }
            
            self.financial_terms = {
                'ипо': 0.3, 'размещение': 0.3, 'листинг': 0.3, 'делистинг': -0.3,
                'отчетность': 0.0, 'мсфо': 0.0, 'рсбу': 0.0, 'квартал': 0.0,
                'дивиденды': 0.4, 'акция': 0.1, 'облигация': 0.1, 'евробонд': 0.1,
                'IPO': 0.4
            }
                        
            self.time_trends = {
                'цифровизация': 0.6, 'искусственный интеллект': 0.7, 'цифровой рубль': 0.4,
                'криптовалюта': 0.3, 'финтех': 0.5, 'биотехнологии': 0.6, 'зеленая энергетика': 0.5,
                'углеродный след': -0.3, 'углеродный налог': -0.5, 'климатическая повестка': -0.2,
                'технологический суверенитет': 0.7,
                'IPO': 0.4
            }
            
            self.industry_dict = {
                'нефтегазовый': {'экспорт': 0.7, 'добыча': 0.6, 'запасы': 0.5},
                'банковский': {'прибыль': 0.8, 'активы': 0.6, 'кредиты': 0.5},
                'металлургия': {'экспорт': 0.7, 'производство': 0.6, 'дивиденды': 0.7},
                'телеком': {'абоненты': 0.6, 'arpu': 0.7, 'покрытие': 0.5},
                'ритейл': {'выручка': 0.7, 'магазины': 0.5, 'like-for-like': 0.6},
                'энергетический': {'газ': 0.6, 'нефть': 0.6}
            }
            
            self.company_dict = {
                'gazp': {'северный поток': 0.0, 'сила сибири': 0.7, 'турецкий поток': 0.6, 'доставка': 0.5},
                'rosn': {'восток ойл': 0.8, 'ванкор': 0.6, 'нефть': 0.8},
                'sber': {'экосистема': 0.7, 'цифровизация': 0.6, 'ипотека': -0.2},
                'rual': {'алюминий': 0.6, 'en+': 0.5, 'металл': 0.6},
                'vtbr': {'розничный портфель': 0.6, 'государственная поддержка': 0.7},
                'nvtk': {'газ': 0.5},
                'gmkn': {'норильский никель': 0.7},
                'tatn': {'нефть': 0.6},
                'mtss': {'телеком': 0.6},
                'alrs': {'алюминий': 0.6},
                'sngs': {'газ': 0.7},
                'mvid': {'видео': 0.5},
                'phor': {'удобрения': 0.5},
                'sibn': {'нефть': 0.6, 'банковский': 0.4},
                'afks': {'система': 0.5, 'функционирование': 0.5},
                'magn': {'металлургия': 0.6},
                'rual': {'металл': 0.6},
}

            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                import pymorphy2
                self.morph = pymorphy2.MorphAnalyzer()
                self.use_morph = True
            except ImportError:
                self.use_morph = False
                print("pymorphy2 не установлен. Используем упрощенный анализ морфологии.")
            
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(max_features=1000)
                self.use_tfidf = True
            except ImportError:
                self.use_tfidf = False
                print("sklearn не установлен. Анализ TF-IDF не будет использоваться.")
    
    def _normalize_russian_word(self, word):
        if self.use_morph:
            return self.morph.parse(word)[0].normal_form
        else:
            suffixes = ['ой', 'ый', 'ая', 'ое', 'ые', 'ого', 'ому', 'ыми', 'ыми', 'ом', 'ям', 'ами', 'а', 'и', 'о', 'у', 'е', 'ы']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 3:
                    return word[:-len(suffix)]
            return word
    
    def _check_contextual_phrase(self, text, context, action):
        if context in text.lower() and action in text.lower():
            return True
        return False
    
    def analyze_text_vader(self, text):
        if not isinstance(text, str) or not text:
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
        
        return self.sid.polarity_scores(text)
    
    def analyze_text_simple_ru(self, text):
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
    
    def analyze_text_advanced_ru(self, text, ticker=None):
        if not isinstance(text, str) or not text:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
        
        text = text.lower()
        
        tokens = word_tokenize(text, language='russian')
        tokens = [word for word in tokens if word not in punctuation]
        
        if self.use_morph:
            normalized_tokens = [self._normalize_russian_word(word) for word in tokens]
        else:
            normalized_tokens = tokens
        
        pos_score = 0
        neg_score = 0
        
        for token in normalized_tokens:
            if token in self.positive_words:
                pos_score += self.positive_words[token]
            if token in self.negative_words:
                neg_score += self.negative_words[token]
            if token in self.financial_terms:
                pos_score += max(0, self.financial_terms[token])
                neg_score += abs(min(0, self.financial_terms[token]))
            if token in self.time_trends:
                pos_score += max(0, self.time_trends[token])
                neg_score += abs(min(0, self.time_trends[token]))
        
        for context, actions in self.context_dict.items():
            for action, score in actions.items():
                if self._check_contextual_phrase(text, context, action):
                    if score > 0:
                        pos_score += score
                    else:
                        neg_score += abs(score)
        
        for industry, terms in self.industry_dict.items():
            if industry in text:
                for term, score in terms.items():
                    if term in text:
                        pos_score += score
        
        if ticker and ticker.lower() in self.company_dict:
            for project, score in self.company_dict[ticker.lower()].items():
                if project in text:
                    pos_score += score
        
        special_cases = {
            'санкции введены': -0.8,
            'санкции отменены': 0.9,
            'сняты ограничения': 0.8,
            'ключевая ставка снижена': 0.7,
            'ключевая ставка повышена': -0.6,
            'импортозамещение успешно': 0.7,
            'снижение экспорта': -0.7,
            'рост экспорта': 0.7,
            'геополитическая напряженность': -0.7,
            'налоговый маневр': -0.5,
            'цены на нефть растут': 0.7,
            'цены на нефть падают': -0.7,
            'дивиденды выросли': 0.8,
            'дивиденды сократились': -0.8,
            'инфляция растет': -0.7,
            'инфляция снижается': 0.7,
            'акции выросли': 0.7,
            'акции упали': -0.7,
            'аномальный объем': -0.6,
            'рекордный рост': 0.8
        }
        
        for case, score in special_cases.items():
            if case in text:
                if score > 0:
                    pos_score += score
                else:
                    neg_score += abs(score)
        
        total_words = len(normalized_tokens)
        if total_words == 0:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0}
        
        pos_ratio = min(1.0, pos_score / (total_words * 0.3))
        neg_ratio = min(1.0, neg_score / (total_words * 0.3))
        
        length_factor = min(1.0, 100 / total_words) if total_words > 0 else 0
        pos_ratio *= (1 + 0.5 * length_factor)
        neg_ratio *= (1 + 0.5 * length_factor)
        
        pos_ratio = min(1.0, pos_ratio)
        neg_ratio = min(1.0, neg_ratio)
        neu_ratio = max(0.0, 1.0 - (pos_ratio + neg_ratio))
        
        compound = (pos_ratio - neg_ratio) * 1.5
        compound = max(-1, min(1, compound))
        
        if 0 < compound < 0.05:
            compound = 0.05
        elif -0.05 < compound < 0:
            compound = -0.05
            
        return {
            'compound': compound,
            'neg': neg_ratio,
            'neu': neu_ratio,
            'pos': pos_ratio
        }
    
    def analyze_stock_impact(self, text, ticker=None):
        strong_indicators = {
            'отчетность превысила ожидания': 0.8,
            'отчетность ниже ожиданий': -0.8,
            'превысила прогноз': 0.7,
            'ниже прогноза': -0.7,
            'рекордная прибыль': 0.9,
            'чистый убыток': -0.8,
            'дивиденды выше': 0.8,
            'дивиденды ниже': -0.8,
            'новый контракт': 0.6,
            'потеря контракта': -0.7,
            'крупная сделка': 0.7,
            'судебное разбирательство': -0.6,
            'штраф': -0.7,
            'кредитный рейтинг повышен': 0.7,
            'кредитный рейтинг понижен': -0.7,
            'реструктуризация долга': -0.5,
            'успешная реструктуризация': 0.6,
            'новые месторождения': 0.8,
            'снижение запасов': -0.7,
            'смена руководства': 0.0,
            'новый генеральный директор': 0.2,
            'отставка руководителя': -0.4
        }
        
        text_lower = text.lower()
        impact = 0.0
        
        for indicator, score in strong_indicators.items():
            if indicator in text_lower:
                impact += score
        
        sentiment = self.analyze_text_advanced_ru(text, ticker)
        sentiment_weight = 0.5
        
        industry_factor = 1.0
        if ticker and ticker.lower() in self.company_dict:
            impact *= 1.2
            
            for project in self.company_dict[ticker.lower()]:
                if project in text_lower:
                    impact += 0.2
        
        final_impact = (impact + sentiment['compound'] * sentiment_weight) * industry_factor
        final_impact = max(-1.0, min(1.0, final_impact))
        
        return final_impact
    
    def analyze_text(self, text, ticker=None):
        if self.use_vader and self.language == 'english':
            return self.analyze_text_vader(text)
        elif self.language == 'russian':
            results = self.analyze_text_advanced_ru(text, ticker)
            
            impact = self.analyze_stock_impact(text, ticker)
            results['stock_impact'] = impact
            
            return results
        else:
            return {'compound': 0, 'neg': 0, 'neu': 1, 'pos': 0, 'stock_impact': 0}
    
    def analyze_news_dataframe(self, df, text_column='clean_text'):
        if df.empty or text_column not in df.columns:
            print(f"DataFrame пуст или не содержит колонку {text_column}")
            return df
        
        result_df = df.copy()
        
        ticker_column = None
        for possible_col in ['ticker', 'symbol', 'stock']:
            if possible_col in result_df.columns:
                ticker_column = possible_col
                break
                
        if ticker_column:
            sentiments = []
            for _, row in result_df.iterrows():
                sentiment = self.analyze_text(row[text_column], ticker=row[ticker_column])
                sentiments.append(sentiment)
        else:
            sentiments = result_df[text_column].apply(self.analyze_text)
        
        result_df['sentiment_compound'] = [x['compound'] for x in sentiments]
        result_df['sentiment_negative'] = [x['neg'] for x in sentiments]
        result_df['sentiment_neutral'] = [x['neu'] for x in sentiments]
        result_df['sentiment_positive'] = [x['pos'] for x in sentiments]
        
        if 'stock_impact' in sentiments[0]:
            result_df['stock_impact'] = [x.get('stock_impact', 0) for x in sentiments]
        
        result_df['sentiment_category'] = pd.cut(
            result_df['sentiment_compound'],
            bins=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            labels=['very negative', 'negative', 'neutral', 'positive', 'very positive']
        )
        
        result_df['sentiment_simple'] = result_df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        return result_df
    
    def analyze_ticker_news(self, news_df, save_path=None):
        analyzed_df = self.analyze_news_dataframe(news_df)
        
        if save_path is not None:
            analyzed_df.to_csv(save_path, index=False)
            print(f"Результаты анализа настроений сохранены в {save_path}")
        
        return analyzed_df
    
    def set_cbr_meetings(self, meeting_dates):
        self.cbr_meetings = meeting_dates
        print(f"Установлено {len(meeting_dates)} дат заседаний ЦБ РФ")
    
    def create_daily_sentiment_series(self, analyzed_df):
        if 'date' not in analyzed_df.columns or 'sentiment_compound' not in analyzed_df.columns:
            print("DataFrame не содержит необходимые колонки (date, sentiment_compound)")
            return pd.DataFrame()
        
        analyzed_df['date'] = pd.to_datetime(analyzed_df['date'])
        
        agg_dict = {
            'sentiment_compound': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'sentiment_positive': ['mean', 'sum'],
            'sentiment_negative': ['mean', 'sum'],
            'sentiment_neutral': ['mean'],
            'id': 'count'
        }
        
        if 'stock_impact' in analyzed_df.columns:
            agg_dict['stock_impact'] = ['mean', 'median', 'std']
        
        daily_sentiment = analyzed_df.groupby(pd.Grouper(key='date', freq='D')).agg(agg_dict)
        
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        
        daily_sentiment = daily_sentiment.rename(columns={
            'sentiment_compound_mean': 'avg_sentiment',
            'id_count': 'news_count'
        })
        
        if 'sentiment_compound_count' in daily_sentiment.columns and daily_sentiment['sentiment_compound_count'].sum() > 0:
            daily_sentiment['positive_ratio'] = daily_sentiment['sentiment_positive_sum'] / daily_sentiment['sentiment_compound_count']
            daily_sentiment['negative_ratio'] = daily_sentiment['sentiment_negative_sum'] / daily_sentiment['sentiment_compound_count']
            daily_sentiment['sentiment_ratio'] = daily_sentiment['positive_ratio'] / daily_sentiment['negative_ratio'].replace(0, 0.001)
        
        daily_sentiment['sentiment_strength'] = daily_sentiment['avg_sentiment'].abs() * np.log1p(daily_sentiment['news_count'])
        
        daily_sentiment['sentiment_direction'] = np.sign(daily_sentiment['avg_sentiment'])
        
        daily_sentiment = daily_sentiment.reset_index()
        if not daily_sentiment.empty:
            date_range = pd.date_range(start=daily_sentiment['date'].min(), end=daily_sentiment['date'].max(), freq='D')
            daily_sentiment = daily_sentiment.set_index('date').reindex(date_range).fillna(0).reset_index()
            daily_sentiment.rename(columns={'index': 'date'}, inplace=True)
            
        daily_sentiment = self._integrate_cbr_meetings(daily_sentiment)
        
        daily_sentiment['day_of_week'] = daily_sentiment['date'].dt.dayofweek
        daily_sentiment['month'] = daily_sentiment['date'].dt.month
        daily_sentiment['is_weekend'] = daily_sentiment['day_of_week'].isin([5, 6]).astype(int)
        
        daily_sentiment['trading_activity_index'] = 1.0
        daily_sentiment.loc[daily_sentiment['is_weekend'] == 1, 'trading_activity_index'] = 0.3
        
        daily_sentiment['weighted_sentiment'] = daily_sentiment['avg_sentiment'] * daily_sentiment['trading_activity_index']
        
        return daily_sentiment
    
    def _integrate_cbr_meetings(self, daily_sentiment_df):
        daily_sentiment_df['cbr_meeting'] = 0
        daily_sentiment_df['days_to_cbr_meeting'] = 99
        
        for meeting_date in self.cbr_meetings:
            meeting_dt = pd.to_datetime(meeting_date).date()
            mask = daily_sentiment_df['date'].dt.date == meeting_dt
            daily_sentiment_df.loc[mask, 'cbr_meeting'] = 1
            
            for days_before in range(1, 6):
                pre_meeting_dt = meeting_dt - pd.Timedelta(days=days_before)
                pre_mask = daily_sentiment_df['date'].dt.date == pre_meeting_dt
                daily_sentiment_df.loc[pre_mask, 'days_to_cbr_meeting'] = days_before
        
        daily_sentiment_df['cbr_proximity_factor'] = 1.0
        daily_sentiment_df.loc[daily_sentiment_df['cbr_meeting'] == 1, 'cbr_proximity_factor'] = 1.5
        for days in range(1, 6):
            factor = 1.0 + (6 - days) * 0.1
            daily_sentiment_df.loc[daily_sentiment_df['days_to_cbr_meeting'] == days, 'cbr_proximity_factor'] = factor
        
        daily_sentiment_df['cbr_adjusted_sentiment'] = daily_sentiment_df['avg_sentiment'] * daily_sentiment_df['cbr_proximity_factor']
        
        return daily_sentiment_df

