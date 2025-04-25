import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import traceback

from pys.utils.logger import BaseLogger
from pys.data_collection.kbd_analyzer import KBDAnalyzer

class BondStrategy(ABC, BaseLogger):
    """
    Абстрактный базовый класс для стратегий выбора облигаций
    """
    def __init__(self, name):
        super().__init__(f'Strategy_{name}')
        self.name = name
        
    @abstractmethod
    def get_filter_params(self):
        """Получить параметры фильтрации облигаций"""
        pass
    
    @abstractmethod
    def get_portfolio_params(self):
        """Получить параметры формирования портфеля"""
        pass
    
    @abstractmethod
    def rank_bonds(self, bonds_df):
        """Ранжировать облигации по предпочтительности"""
        pass

class KBDRateExpectationStrategy(BondStrategy):
    """
    Стратегия на основе ожиданий по процентным ставкам,
    определяемым из кривой бескупонной доходности
    """
    def __init__(self, kbd_analyzer=None, risk_tolerance='medium'):
        """
        Инициализация стратегии
        
        Args:
            kbd_analyzer: анализатор КБД или None для создания нового
            risk_tolerance: толерантность к риску ('low', 'medium', 'high')
        """
        super().__init__('KBD_RateExpectation')
        self.risk_tolerance = risk_tolerance
        self.kbd_analyzer = kbd_analyzer or KBDAnalyzer()
        self.kbd_metrics = self.kbd_analyzer.get_latest_kbd_metrics()
        self.market_state = self.kbd_metrics.get('market_state', 'neutral')
        
        self.logger.info(f"Инициализирована стратегия с параметрами: толерантность к риску={risk_tolerance}, состояние рынка={self.market_state}")
    
    def get_filter_params(self):
        """Получить параметры фильтрации облигаций на основе КБД"""
        base_params = self.kbd_analyzer.get_bonds_recommendations()
        
        # Модифицируем параметры в зависимости от толерантности к риску
        if self.risk_tolerance == 'low':
            # Более консервативный подход
            base_params['max_duration'] = min(base_params['max_duration'], 3.0)
            base_params['min_yield'] = max(base_params['min_yield'], 7.0)
        elif self.risk_tolerance == 'high':
            # Более агрессивный подход
            base_params['max_duration'] = max(base_params['max_duration'], 5.0)
            base_params['min_yield'] = max(base_params['min_yield'] - 1.0, 5.0)
        
        # Корректировка при отсутствии данных
        if base_params.get('market_condition') == 'unknown':
            # Используем широкие параметры, если состояние рынка неизвестно
            base_params.update({
                'min_yield': 6.0,
                'max_yield': 22.0,
                'min_duration': 0.1,
                'max_duration': 5.0
            })
        
        self.logger.info(f"Параметры фильтрации: {base_params}")
        return base_params
    
    def get_portfolio_params(self):
        """Получить параметры формирования портфеля"""
        # Параметры зависят от состояния рынка и толерантности к риску
        if self.market_state == 'inverted':
            # Инвертированная кривая (риск рецессии)
            n_bonds = 7 if self.risk_tolerance == 'low' else 5
            weighting = 'bond_score' if self.risk_tolerance == 'high' else 'inverse_duration'
        elif self.market_state == 'flat':
            # Плоская кривая
            n_bonds = 5
            weighting = 'equal'
        elif self.market_state == 'steep':
            # Крутая кривая (ожидание роста)
            n_bonds = 3 if self.risk_tolerance == 'high' else 5
            weighting = 'yield' if self.risk_tolerance == 'high' else 'bond_score'
        else:
            # Нормальная кривая или неизвестное состояние
            n_bonds = 5
            weighting = 'inverse_duration'
        
        params = {
            'n_bonds': n_bonds,
            'weighting_strategy': weighting,
            'portfolio_stability': 0.7 if self.risk_tolerance == 'medium' else (0.8 if self.risk_tolerance == 'low' else 0.5)
        }
        
        self.logger.info(f"Параметры формирования портфеля: {params}")
        return params
    
    def rank_bonds(self, bonds_df):
        """
        Ранжировать облигации на основе КБД и толерантности к риску
        
        Args:
            bonds_df: DataFrame с облигациями
            
        Returns:
            DataFrame с добавленным столбцом стратегического ранга
        """
        if bonds_df is None or len(bonds_df) == 0:
            return bonds_df
        
        df = bonds_df.copy()
        
        # Получаем ключевые показатели КБД для разных сроков
        kbd_values = self.kbd_metrics.get('kbd_values', {})
        
        try:
            # Рассчитываем спред к КБД для каждой облигации
            if 'duration_years' in df.columns and 'yield' in df.columns:
                df['kbd_spread'] = df.apply(
                    lambda row: self._calculate_spread_to_kbd(
                        row['yield'], row['duration_years'], kbd_values
                    ),
                    axis=1
                )
                
                # Рассчитываем ранг на основе спреда и толерантности к риску
                if self.risk_tolerance == 'low':
                    # Консервативный подход: предпочитаем более низкую дюрацию и средний спред
                    df['strategy_rank'] = (
                        df['kbd_spread'] * 0.7 - 
                        df['duration_years'] * 0.3
                    )
                elif self.risk_tolerance == 'high':
                    # Агрессивный подход: предпочитаем высокий спред
                    df['strategy_rank'] = (
                        df['kbd_spread'] * 0.9 + 
                        df['duration_years'] * 0.1
                    )
                else:
                    # Сбалансированный подход
                    df['strategy_rank'] = df['kbd_spread']
                
                # Нормализуем ранг
                if not df['strategy_rank'].isna().all():
                    min_rank = df['strategy_rank'].min()
                    max_rank = df['strategy_rank'].max()
                    if max_rank > min_rank:
                        df['strategy_rank'] = (df['strategy_rank'] - min_rank) / (max_rank - min_rank)
                
                self.logger.info(f"Выполнено ранжирование {len(df)} облигаций")
            else:
                self.logger.warning("Отсутствуют нужные колонки для ранжирования")
                df['strategy_rank'] = 0.5
        
        except Exception as e:
            self.logger.error(f"Ошибка при ранжировании облигаций: {e}\n{traceback.format_exc()}")
            df['strategy_rank'] = 0.5
            
        return df
    
    def _calculate_spread_to_kbd(self, bond_yield, duration, kbd_values):
        """
        Рассчитать спред доходности облигации к КБД
        
        Args:
            bond_yield: доходность облигации
            duration: дюрация облигации
            kbd_values: значения КБД для разных сроков
            
        Returns:
            float: спред доходности
        """
        # Находим ближайшие точки на кривой КБД
        tenors = sorted([(float(k.replace('Y', '')), v) for k, v in kbd_values.items() if not pd.isna(v)], key=lambda x: x[0])
        
        if not tenors:
            return 0
            
        # Если дюрация меньше минимального тенора
        if duration <= tenors[0][0]:
            return bond_yield - tenors[0][1]
            
        # Если дюрация больше максимального тенора
        if duration >= tenors[-1][0]:
            return bond_yield - tenors[-1][1]
            
        # Линейная интерполяция
        for i in range(len(tenors) - 1):
            tenor1, yield1 = tenors[i]
            tenor2, yield2 = tenors[i + 1]
            
            if tenor1 <= duration <= tenor2:
                # Интерполируем доходность КБД для данной дюрации
                interpolated_yield = yield1 + (yield2 - yield1) * (duration - tenor1) / (tenor2 - tenor1)
                return bond_yield - interpolated_yield
                
        return 0
