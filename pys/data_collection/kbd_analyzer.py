import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import traceback

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH
from pys.data_collection.kbd import KBDDownloader

class KBDAnalyzer(BaseLogger):
    """
    Класс для анализа кривой бескупонной доходности (КБД) и
    определения оптимальных параметров для выбора облигаций
    """
    
    def __init__(self, kbd_data=None, output_dir=f'{BASE_PATH}/data/processed_data/BONDS/kbd'):
        """
        Инициализация анализатора КБД
        
        Args:
            kbd_data: DataFrame с данными КБД или None для автоматической загрузки
            output_dir: Директория для сохранения результатов
        """
        super().__init__('KBDAnalyzer')
        self.output_dir = output_dir
        
        # Создаем директории для разных типов данных
        self.viz_dir = os.path.join(self.output_dir, 'viz')
        self.analysis_dir = os.path.join(self.output_dir, 'analysis')
        
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        if kbd_data is not None:
            self.kbd_data = kbd_data
        else:
            downloader = KBDDownloader(output_dir=output_dir)
            self.kbd_data = downloader.load_kbd_data()
            
            if self.kbd_data is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                self.kbd_data = downloader.get_kbd(start_date, end_date)
        
        if self.kbd_data is not None:
            self.kbd_data = self._preprocess_kbd_data()
            self.logger.info(f"КБД данные загружены, {len(self.kbd_data)} записей")
        else:
            self.logger.warning("Не удалось загрузить данные КБД")
        
    def _preprocess_kbd_data(self):
        """Предобработка данных КБД"""
        df = self.kbd_data.copy()
        
        df['date'] = pd.to_datetime(df['date'])
        
        tenor_columns = ['0.25Y', '0.5Y', '0.75Y', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
        
        for col in tenor_columns:
            if col in df.columns:
                if df[col].dtype == object:  # Если колонка строкового типа
                    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Добавляем наклон кривой (разница между доходностью на разных сроках)
        if '10Y' in df.columns and '2Y' in df.columns:
            df['slope_10y_2y'] = df['10Y'] - df['2Y']
        
        if '5Y' in df.columns and '1Y' in df.columns:
            df['slope_5y_1y'] = df['5Y'] - df['1Y']
        
        return df

    def visualize_kbd(self, save_path=None):
        """
        Визуализировать текущую КБД
        
        Args:
            save_path: путь для сохранения визуализации
        
        Returns:
            str: путь к сохраненному файлу
        """
        if self.kbd_data is None or len(self.kbd_data) == 0:
            self.logger.warning("Нет данных КБД для визуализации")
            return None
            
        try:
            # Получаем последнюю запись КБД
            latest_kbd = self.kbd_data.iloc[-1]
            date_str = latest_kbd['date'].strftime('%Y-%m-%d') if hasattr(latest_kbd['date'], 'strftime') else str(latest_kbd['date'])
            
            # Получаем точки для построения кривой
            tenors = []
            yields = []
            
            # Стандартизированные имена колонок для тенора
            tenor_columns = ['0.25Y', '0.5Y', '0.75Y', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
            
            for col in tenor_columns:
                if col in latest_kbd and not pd.isna(latest_kbd[col]):
                    tenor = float(col.replace('Y', ''))
                    tenors.append(tenor)
                    yields.append(latest_kbd[col])
            
            if not tenors:
                self.logger.warning("Недостаточно данных для построения кривой")
                return None
                
            plt.figure(figsize=(10, 6))
            plt.plot(tenors, yields, 'o-', linewidth=2)
            plt.title(f'Кривая бескупонной доходности на {date_str}')
            plt.xlabel('Срок до погашения (лет)')
            plt.ylabel('Доходность (%)')
            plt.grid(True, alpha=0.3)
            
            plt.fill_between(tenors, yields, alpha=0.2)
            
            for i, (x, y) in enumerate(zip(tenors, yields)):
                if x in [1, 2, 5, 10]:
                    plt.annotate(
                        f'{y:.2f}%',
                        (x, y),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center'
                    )
            
            if save_path is None:
                save_path = os.path.join(self.viz_dir, f'kbd_curve_{datetime.now().strftime("%Y%m%d")}.png')
            else:
                if not os.path.dirname(save_path) == self.viz_dir:
                    save_path = os.path.join(self.viz_dir, os.path.basename(save_path))
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"КБД визуализация сохранена в {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации КБД: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def get_latest_kbd_metrics(self):
        """
        Получить последние метрики КБД для формирования рекомендаций
        
        Returns:
            dict: Словарь с метриками КБД
        """
        if self.kbd_data is None or len(self.kbd_data) == 0:
            return {}
        
        latest_kbd = self.kbd_data.iloc[-1]
        
        # Собираем доступные значения КБД для всех тенорів
        tenor_columns = ['0.5Y', '0.75Y', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
        available_tenors = {}
        
        for col in tenor_columns:
            if col in latest_kbd.index and not pd.isna(latest_kbd[col]):
                available_tenors[col] = latest_kbd[col]
            elif col in latest_kbd.keys() and not pd.isna(latest_kbd[col]):
                available_tenors[col] = latest_kbd[col]
        
        metrics = {
            'date': latest_kbd['date'],
            'market_state': self._determine_market_state(latest_kbd),
            'optimal_duration': self._get_optimal_duration(latest_kbd),
            'yield_threshold': self._get_yield_threshold(latest_kbd),
            'kbd_values': available_tenors
        }
        
        self.logger.info(f"Сформированы метрики КБД: {metrics}")
        return metrics

    
    def _determine_market_state(self, latest_kbd):
        """
        Определить состояние рынка на основе формы кривой
        
        Returns:
            str: 'flat', 'normal', 'inverted', 'steep'
        """
        if 'slope_10y_2y' in latest_kbd.index and not pd.isna(latest_kbd['slope_10y_2y']):
            slope = latest_kbd['slope_10y_2y']
            
            if abs(slope) < 0.3:
                return 'flat'  # Плоская кривая
            elif slope < -0.3:
                return 'inverted'  # Инвертированная кривая (риск рецессии)
            elif slope > 1.5:
                return 'steep'  # Крутая кривая (ожидание роста)
            else:
                return 'normal'  # Нормальная кривая
        
        return 'unknown'
    
    def _get_optimal_duration(self, latest_kbd):
        """
        Определить оптимальную дюрацию на основе формы кривой
        
        Returns:
            tuple: (min_duration, max_duration)
        """
        market_state = self._determine_market_state(latest_kbd)
        
        if market_state == 'inverted':
            # При инвертированной кривой лучше короткая дюрация
            return (0.1, 2.0)
        elif market_state == 'flat':
            # При плоской кривой - сбалансированный подход
            return (0.5, 3.0)
        elif market_state == 'steep':
            # При крутой кривой выгоднее длинная дюрация
            return (2.0, 5.0)
        else:
            # В обычных условиях - умеренный диапазон
            return (0.5, 4.0)
    
    def _get_yield_threshold(self, latest_kbd):
        """
        Определить минимальный порог доходности для фильтрации облигаций
        
        Returns:
            float: минимальная доходность
        """
        # Базовый подход: минимальная доходность = КБД на 1 год + премия
        base_yield = latest_kbd.get('1Y', 0)
        
        # Если нет данных по КБД, используем базовый порог
        if pd.isna(base_yield) or base_yield == 0:
            return 6.0
            
        # Добавляем премию в зависимости от состояния рынка
        market_state = self._determine_market_state(latest_kbd)
        
        if market_state == 'inverted':
            premium = 3.0  # Высокая премия в условиях инверсии
        elif market_state == 'flat':
            premium = 2.0  # Средняя премия
        else:
            premium = 1.5  # Обычная премия
            
        return base_yield + premium
    
    def get_bonds_recommendations(self):
        """
        Получить рекомендации для фильтрации облигаций на основе КБД
        
        Returns:
            dict: параметры для фильтрации облигаций
        """
        kbd_metrics = self.get_latest_kbd_metrics()
        
        if not kbd_metrics:
            # Стандартные параметры, если нет данных КБД
            return {
                'min_yield': 6.0,
                'max_yield': 22.0,
                'min_duration': 0.1,
                'max_duration': 5.0,
                'market_condition': 'unknown'
            }
        
        min_duration, max_duration = kbd_metrics.get('optimal_duration', (0.1, 5.0))
        min_yield = kbd_metrics.get('yield_threshold', 6.0)
        
        # Ограничиваем максимальную доходность
        max_yield = min_yield + 16  # Примерно +16% от минимальной доходности
        
        recommendations = {
            'min_yield': min_yield,
            'max_yield': max_yield,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'market_condition': kbd_metrics.get('market_state', 'unknown')
        }
        
        self.logger.info(f"Сформированы рекомендации для облигаций: {recommendations}")
        return recommendations
    
    def get_optimal_weighting_strategy(self, latest_kbd=None):
        """
        Определить оптимальную стратегию взвешивания на основе формы кривой КБД
        
        Args:
            latest_kbd: последние данные КБД или None для использования внутренних данных
        
        Returns:
            dict: рекомендуемая стратегия взвешивания и параметры
        """
        if latest_kbd is None:
            if self.kbd_data is None or len(self.kbd_data) == 0:
                return {'strategy': 'inverse_duration', 'reason': 'default'}
            latest_kbd = self.kbd_data.iloc[-1]
        
        market_state = self._determine_market_state(latest_kbd)
        
        if market_state == 'inverted':
            # При инвертированной кривой, когда короткие ставки выше длинных,
            # лучше взвешивать по доходности или использовать равное взвешивание
            # для минимизации дюрационного риска
            return {
                'strategy': 'equal',
                'reason': 'inverted_curve',
                'description': 'Равное взвешивание для защиты от инверсии кривой доходности'
            }
        elif market_state == 'flat':
            # При плоской кривой доходности эффект от разных стратегий менее выражен,
            # можно использовать равное взвешивание
            return {
                'strategy': 'equal',
                'reason': 'flat_curve',
                'description': 'Равное взвешивание при плоской кривой доходности'
            }
        elif market_state == 'steep':
            # При крутой кривой выгоднее инвестировать в длинные облигации,
            # поэтому взвешивание по доходности может быть оптимальным
            return {
                'strategy': 'yield',
                'reason': 'steep_curve',
                'description': 'Взвешивание по доходности при крутой кривой доходности'
            }
        else:
            # В обычных рыночных условиях обратное взвешивание по дюрации
            # хорошо балансирует риск и доходность
            return {
                'strategy': 'inverse_duration',
                'reason': 'normal_curve',
                'description': 'Взвешивание обратно пропорционально дюрации при нормальной кривой'
            }
        

    def get_comprehensive_recommendations(self):
        """
        Получить комплексные рекомендации для формирования портфеля облигаций
        на основе анализа КБД
        
        Returns:
            dict: полные рекомендации для формирования портфеля
        """
        # Получаем базовые метрики
        kbd_metrics = self.get_latest_kbd_metrics()
        
        if not kbd_metrics:
            # Стандартные рекомендации при отсутствии данных КБД
            return {
                'min_yield': 6.0,
                'max_yield': 22.0,
                'min_duration': 0.1,
                'max_duration': 5.0,
                'market_condition': 'unknown',
                'weighting_strategy': 'inverse_duration',
                'strategy_reason': 'default'
            }
        
        # Получаем рекомендации по фильтрации
        filter_params = self.get_bonds_recommendations()
        
        # Получаем рекомендации по стратегии взвешивания
        weighting_recommendation = self.get_optimal_weighting_strategy()
        
        # Формируем комплексные рекомендации
        comprehensive_recommendations = filter_params.copy()
        comprehensive_recommendations.update({
            'weighting_strategy': weighting_recommendation['strategy'],
            'strategy_reason': weighting_recommendation['reason'],
            'strategy_description': weighting_recommendation.get('description', '')
        })
        
        self.logger.info(f"Сформированы комплексные рекомендации: {comprehensive_recommendations}")
        return comprehensive_recommendations

