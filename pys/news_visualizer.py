import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class NewsVisualizer:
    """Класс для визуализации новостных данных"""
    
    def __init__(self, output_dir=None):
        """
        Инициализация визуализатора новостей
        
        Args:
            output_dir (str): Директория для сохранения визуализаций
        """
        self.output_dir = output_dir
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_sentiment_time_series(self, sentiment_ts, ticker, save=False):
        """
        Визуализация временного ряда настроений
        
        Args:
            sentiment_ts (pd.DataFrame): DataFrame с временным рядом настроений
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            
        Returns:
            go.Figure: Объект графика Plotly
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=sentiment_ts['date'],
                y=sentiment_ts['avg_sentiment'],
                name="Среднее настроение",
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=sentiment_ts['date'],
                y=sentiment_ts['news_count'],
                name="Количество новостей",
                marker_color='lightblue',
                opacity=0.7
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=f"Динамика настроений в новостях для {ticker}",
            xaxis_title="Дата",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Настроение (-1 до 1)", secondary_y=False)
        fig.update_yaxes(title_text="Количество новостей", secondary_y=True)
        
        if save and self.output_dir:
            file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_time_series.html")
            fig.write_html(file_path)
            print(f"График сохранен в {file_path}")
        
        return fig
    
    def plot_event_distribution(self, df, ticker, save=False):
        """
        Визуализация распределения типов событий
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            
        Returns:
            go.Figure: Объект графика Plotly
        """
        event_columns = [col for col in df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
        
        event_counts = {}
        for col in event_columns:
            event_type = col.replace('event_', '')
            event_counts[event_type] = df[col].sum()
        
        event_df = pd.DataFrame({
            'event_type': list(event_counts.keys()),
            'count': list(event_counts.values())
        })
        
        event_df = event_df.sort_values('count', ascending=False)
        
        fig = px.bar(
            event_df, 
            x='event_type', 
            y='count',
            title=f"Распределение типов событий для {ticker}",
            labels={'event_type': 'Тип события', 'count': 'Количество'}
        )
        
        if save and self.output_dir:
            file_path = os.path.join(self.output_dir, f"{ticker}_event_distribution.html")
            fig.write_html(file_path)
            print(f"График сохранен в {file_path}")
        
        return fig
    
    def plot_sentiment_by_event_type(self, df, ticker, save=False):
        """
        Визуализация распределения настроений по типам событий
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями и настроениями
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            
        Returns:
            go.Figure: Объект графика Plotly
        """
        event_columns = [col for col in df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
        
        data = []
        
        for col in event_columns:
            event_type = col.replace('event_', '')
  
            sentiment_values = df[df[col]]['sentiment_compound'].values
            if len(sentiment_values) > 0:
                data.append({
                    'event_type': event_type,
                    'sentiments': sentiment_values
                })
        
        if not data:
            print("Нет данных для визуализации настроений по типам событий")
            fig = go.Figure()
            fig.update_layout(title="Нет данных для визуализации")
            return fig
        
        fig = go.Figure()
        
        for item in data:
            fig.add_trace(go.Box(
                y=item['sentiments'],
                name=item['event_type'],
                boxmean=True
            ))
        
        fig.update_layout(
            title=f"Распределение настроений по типам событий для {ticker}",
            yaxis_title="Настроение (-1 до 1)",
            xaxis_title="Тип события",
            boxmode='group'
        )
        
        if save and self.output_dir:
            file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_by_event_type.html")
            fig.write_html(file_path)
            print(f"График сохранен в {file_path}")
        
        return fig
    
    def create_news_dashboard(self, news_df, sentiment_ts, ticker, save=False):
        """
        Создание дашборда с результатами анализа новостей
        
        Args:
            news_df (pd.DataFrame): DataFrame с новостями и результатами анализа
            sentiment_ts (pd.DataFrame): DataFrame с временным рядом настроений
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли дашборд в файл
            
        Returns:
            dict: Словарь с объектами графиков Plotly
        """
        sentiment_fig = self.plot_sentiment_time_series(sentiment_ts, ticker, save=False)
        event_dist_fig = self.plot_event_distribution(news_df, ticker, save=False)
        sentiment_by_event_fig = self.plot_sentiment_by_event_type(news_df, ticker, save=False)
        
        sentiment_hist_fig = px.histogram(
            news_df, 
            x='sentiment_compound',
            title=f"Распределение настроений для {ticker}",
            labels={'sentiment_compound': 'Настроение (-1 до 1)', 'count': 'Количество'},
            nbins=50,
            color_discrete_sequence=['lightblue']
        )
        
        if save and self.output_dir:
            ticker_dir = os.path.join(self.output_dir, ticker)
            if not os.path.exists(ticker_dir):
                os.makedirs(ticker_dir)
            
            sentiment_fig.write_html(os.path.join(ticker_dir, "sentiment_time_series.html"))
            event_dist_fig.write_html(os.path.join(ticker_dir, "event_distribution.html"))
            sentiment_by_event_fig.write_html(os.path.join(ticker_dir, "sentiment_by_event_type.html"))
            sentiment_hist_fig.write_html(os.path.join(ticker_dir, "sentiment_histogram.html"))
            
            print(f"Дашборд сохранен в директории {ticker_dir}")
        
        dashboard = {
            'sentiment_time_series': sentiment_fig,
            'event_distribution': event_dist_fig,
            'sentiment_by_event_type': sentiment_by_event_fig,
            'sentiment_histogram': sentiment_hist_fig
        }
        
        return dashboard
