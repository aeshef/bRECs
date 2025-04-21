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
    
    def plot_sentiment_time_series(self, sentiment_ts, ticker, save=False, use_plotly=True):
        """
        Визуализация временного ряда настроений
        
        Args:
            sentiment_ts (pd.DataFrame): DataFrame с временным рядом настроений
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            use_plotly (bool): Использовать ли Plotly (иначе Matplotlib)
            
        Returns:
            go.Figure или plt.Figure: Объект графика
        """
        if use_plotly:
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
        else:
            # Вариант с matplotlib
            plt.figure(figsize=(15, 8))
            plt.subplot(2, 1, 1)
            plt.plot(sentiment_ts['date'], sentiment_ts['avg_sentiment'], label='Среднее настроение')
            if 'sentiment_ma_7d' in sentiment_ts.columns:
                plt.plot(sentiment_ts['date'], sentiment_ts['sentiment_ma_7d'], label='Скользящее среднее (7 дней)')
            plt.title(f'Динамика настроений для {ticker}')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(sentiment_ts['date'], sentiment_ts['news_count'], label='Количество новостей')
            plt.title(f'Объем новостей для {ticker}')
            plt.legend()
            
            plt.tight_layout()
            
            if save and self.output_dir:
                file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_dynamics.png")
                plt.savefig(file_path)
                print(f"График сохранен в {file_path}")
                plt.close()
            
            fig = plt  # Для совместимости с возвращаемым значением
            
        return fig
    
    def plot_event_distribution(self, df, ticker, save=False, use_plotly=True):
        """
        Визуализация распределения типов событий
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            use_plotly (bool): Использовать ли Plotly (иначе Matplotlib)
            
        Returns:
            go.Figure или plt.Figure: Объект графика
        """
        event_columns = [col for col in df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
        
        event_counts = {}
        for col in event_columns:
            event_type = col.replace('event_', '')
            event_counts[event_type] = df[col].sum()
        
        if use_plotly:
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
        else:
            # Вариант с matplotlib
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(event_counts.keys()), y=list(event_counts.values()))
            plt.title(f'Распределение типов событий для {ticker}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save and self.output_dir:
                file_path = os.path.join(self.output_dir, f"{ticker}_event_distribution.png")
                plt.savefig(file_path)
                print(f"График сохранен в {file_path}")
                # Очищаем текущий график, иначе он будет использоваться в следующей визуализации
                plt.close()
            
            fig = plt  # Для совместимости с возвращаемым значением
            
        return fig
    
    def plot_sentiment_by_event_type(self, df, ticker, save=False, use_plotly=True):
        """
        Визуализация распределения настроений по типам событий
        
        Args:
            df (pd.DataFrame): DataFrame с обнаруженными событиями и настроениями
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            use_plotly (bool): Использовать ли Plotly (иначе Matplotlib)
            
        Returns:
            go.Figure или plt.Figure: Объект графика
        """
        event_columns = [col for col in df.columns if col.startswith('event_') and col not in ['event_impact', 'event_direction', 'has_event']]
        
        data = []
        
        for col in event_columns:
            event_type = col.replace('event_', '')
  
            sentiment_values = df[df[col] == 1]['sentiment_compound'].values
            if len(sentiment_values) > 0:
                data.append({
                    'event_type': event_type,
                    'sentiments': sentiment_values
                })
        
        if not data:
            print("Нет данных для визуализации настроений по типам событий")
            if use_plotly:
                fig = go.Figure()
                fig.update_layout(title="Нет данных для визуализации")
            else:
                plt.figure()
                plt.title("Нет данных для визуализации")
                fig = plt
            return fig
        
        if use_plotly:
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
        else:
            # Вариант с matplotlib
            plt.figure(figsize=(14, 8))
            
            # Создаем DataFrame для seaborn
            plot_data = []
            for item in data:
                for sentiment in item['sentiments']:
                    plot_data.append({
                        'event_type': item['event_type'],
                        'sentiment': sentiment
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Используем boxplot seaborn
            sns.boxplot(x='event_type', y='sentiment', data=plot_df)
            plt.title(f'Распределение настроений по типам событий для {ticker}')
            plt.xlabel('Тип события')
            plt.ylabel('Настроение (-1 до 1)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save and self.output_dir:
                file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_by_event_type.png")
                plt.savefig(file_path)
                print(f"График сохранен в {file_path}")
                plt.close()
            
            fig = plt  # Для совместимости с возвращаемым значением
            
        return fig
    
    def plot_sentiment_distribution(self, df, ticker, save=False, use_plotly=True):
        """
        Визуализация распределения настроений
        
        Args:
            df (pd.DataFrame): DataFrame с настроениями
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли график в файл
            use_plotly (bool): Использовать ли Plotly (иначе Matplotlib)
            
        Returns:
            go.Figure или plt.Figure: Объект графика
        """
        if use_plotly:
            fig = px.histogram(
                df, 
                x='sentiment_compound',
                title=f"Распределение настроений для {ticker}",
                labels={'sentiment_compound': 'Настроение (-1 до 1)', 'count': 'Количество'},
                nbins=50,
                color_discrete_sequence=['lightblue']
            )
            
            if save and self.output_dir:
                file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_histogram.html")
                fig.write_html(file_path)
                print(f"График сохранен в {file_path}")
        else:
            # Вариант с matplotlib
            plt.figure(figsize=(10, 5))
            if 'sentiment_category' in df.columns:
                sns.countplot(x='sentiment_category', data=df)
            else:
                plt.hist(df['sentiment_compound'], bins=50, color='lightblue')
            plt.title(f'Распределение настроений для {ticker}')
            plt.tight_layout()
            
            if save and self.output_dir:
                file_path = os.path.join(self.output_dir, f"{ticker}_sentiment_distribution.png")
                plt.savefig(file_path)
                print(f"График сохранен в {file_path}")
                plt.close()
            
            fig = plt  # Для совместимости с возвращаемым значением
            
        return fig
    
    def create_news_dashboard(self, news_df, sentiment_ts, ticker, save=False, use_plotly=True):
        """
        Создание дашборда с результатами анализа новостей
        
        Args:
            news_df (pd.DataFrame): DataFrame с новостями и результатами анализа
            sentiment_ts (pd.DataFrame): DataFrame с временным рядом настроений
            ticker (str): Тикер компании для заголовка
            save (bool): Сохранять ли дашборд в файл
            use_plotly (bool): Использовать ли Plotly (иначе Matplotlib)
            
        Returns:
            dict: Словарь с объектами графиков
        """
        sentiment_fig = self.plot_sentiment_time_series(sentiment_ts, ticker, save=save, use_plotly=use_plotly)
        event_dist_fig = self.plot_event_distribution(news_df, ticker, save=save, use_plotly=use_plotly)
        sentiment_by_event_fig = self.plot_sentiment_by_event_type(news_df, ticker, save=save, use_plotly=use_plotly)
        sentiment_hist_fig = self.plot_sentiment_distribution(news_df, ticker, save=save, use_plotly=use_plotly)
        
        if save and self.output_dir and use_plotly:
            ticker_dir = os.path.join(self.output_dir, ticker)
            if not os.path.exists(ticker_dir):
                os.makedirs(ticker_dir)
            
            # Эти сохранения актуальны только для Plotly, для matplotlib сохранение уже происходит в самих методах
            if hasattr(sentiment_fig, 'write_html'):
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
    
    # Добавляем методы для совместимости с существующим пайплайном
    def visualize_sentiment(self, ticker, news_with_sentiment, daily_sentiment, output_dir):
        """
        Визуализация настроений для пайплайна новостей
        
        Args:
            ticker (str): Тикер компании
            news_with_sentiment (pd.DataFrame): DataFrame с новостями и настроениями
            daily_sentiment (pd.DataFrame): DataFrame с ежедневными настроениями
            output_dir (str): Директория для сохранения
        """
        self.output_dir = output_dir
        
        # Гистограмма распределения настроений
        self.plot_sentiment_distribution(news_with_sentiment, ticker, save=True, use_plotly=False)
        
        # Временной ряд настроений
        self.plot_sentiment_time_series(daily_sentiment, ticker, save=True, use_plotly=False)
    
    def visualize_events(self, ticker, news_with_events, daily_events, output_dir):
        """
        Визуализация событий для пайплайна новостей
        
        Args:
            ticker (str): Тикер компании
            news_with_events (pd.DataFrame): DataFrame с новостями и событиями
            daily_events (pd.DataFrame): DataFrame с ежедневными событиями
            output_dir (str): Директория для сохранения
        """
        self.output_dir = output_dir
        
        # Распределение типов событий
        self.plot_event_distribution(news_with_events, ticker, save=True, use_plotly=False)
        
        # Настроения по типам событий (дополнительно)
        self.plot_sentiment_by_event_type(news_with_events, ticker, save=True, use_plotly=False)
