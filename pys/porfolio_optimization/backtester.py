import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import sys

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class Backtester(BaseLogger):
    def __init__(self, input_file=None, portfolio_weights=None, start_date=None, end_date=None, 
                 log_level=logging.INFO, bonds_data_dir=None):
        """
        Бэктестер для оценки эффективности стратегии с поддержкой полного портфеля
        
        Parameters:
        -----------
        input_file : str, optional
            Путь к файлу с данными и сигналами
        portfolio_weights : dict, optional
            Словарь с весами активов в портфеле
        start_date : str, optional
            Дата начала бэктеста в формате 'YYYY-MM-DD'
        end_date : str, optional
            Дата окончания бэктеста в формате 'YYYY-MM-DD'
        bonds_data_dir : str, optional
            Директория с историческими данными по облигациям
        """
        super().__init__('Backtester')
        self.input_file = input_file
        self.portfolio_weights = portfolio_weights
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.bonds_data_dir = bonds_data_dir
        
        self.df = None
        self.bonds_df = {}
        self.returns_by_ticker = {}
        self.bonds_returns = {}
        self.portfolio_returns = None
        self.metrics = None
        
        # Извлекаем детали по облигациям, если они есть
        self.bonds_details = None
        if portfolio_weights and 'rf_details' in portfolio_weights:
            self.bonds_details = portfolio_weights['rf_details']
            self.logger.info(f"Найдены детали по {len(self.bonds_details)} облигациям в портфеле")
    
    def load_data(self, input_file=None):
        """Загрузка данных для бэктестирования акций"""
        if input_file:
            self.input_file = input_file
                
        if self.input_file:
            self.logger.info(f"Загрузка данных по акциям из {self.input_file}")
                
            try:
                self.df = pd.read_csv(self.input_file)
                    
                # Преобразование даты в индекс, если есть
                if 'date' in self.df.columns:
                    self.df['date'] = pd.to_datetime(self.df['date'])
                    self.df.set_index('date', inplace=True)
                        
                self.logger.info(f"Загружено {len(self.df)} строк данных по акциям")
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке данных по акциям: {e}")
                return None
        else:
            self.logger.error("Не указан источник данных по акциям")
            return None
                
        # Фильтрация по дате
        if self.start_date:
            self.df = self.df[self.df.index >= self.start_date]
        if self.end_date:
            self.df = self.df[self.df.index <= self.end_date]
                
        if self.df is not None and len(self.df) > 0:
            self.logger.info(f"Период бэктеста: {self.df.index.min().strftime('%Y-%m-%d')} - {self.df.index.max().strftime('%Y-%m-%d')}")
                
        return self.df
    
    def load_bonds_data(self):
        """Загрузка исторических данных по облигациям"""
        if not self.bonds_details:
            self.logger.info("Нет данных о составе портфеля облигаций")
            return None
        
        self.logger.info("Загрузка исторических данных по облигациям")
        
        for bond_name, details in self.bonds_details.items():
            security_code = details['security_code']
            
            # Проверяем, существуют ли исторические данные
            bond_file = None
            if self.bonds_data_dir:
                bond_file = f"{self.bonds_data_dir}/{security_code}.csv"
                if not os.path.exists(bond_file):
                    bond_file = None
            
            if bond_file:
                try:
                    # Загружаем исторические данные
                    bond_df = pd.read_csv(bond_file)
                    bond_df['date'] = pd.to_datetime(bond_df['date'])
                    bond_df.set_index('date', inplace=True)
                    
                    # Фильтрация по дате
                    if self.start_date:
                        bond_df = bond_df[bond_df.index >= self.start_date]
                    if self.end_date:
                        bond_df = bond_df[bond_df.index <= self.end_date]
                    
                    self.bonds_df[security_code] = bond_df
                    self.logger.info(f"Загружены исторические данные для облигации {security_code} ({bond_name})")
                except Exception as e:
                    self.logger.warning(f"Ошибка при загрузке данных для облигации {security_code}: {e}")
            else:
                # Если исторические данные недоступны, используем доходность из details
                self.logger.info(f"Исторические данные для облигации {security_code} недоступны, " 
                               f"будет использована фиксированная доходность {details.get('yield', 0):.2f}%")
        
        return self.bonds_df
    
    def calculate_returns(self, df=None, signals_col='signal', price_col='close'):
        """
        Рассчитывает доходность на основе сигналов с учетом коротких позиций
        """
        if df is None:
            df = self.df
            
        if df is None:
            self.logger.error("Данные не загружены")
            return None
            
        if signals_col not in df.columns:
            self.logger.error(f"Колонка {signals_col} не найдена")
            return None
            
        if price_col not in df.columns:
            self.logger.error(f"Колонка {price_col} не найдена")
            return None
        
        # Копируем DataFrame
        df_returns = df.copy()
        
        # Рассчитываем дневную доходность
        df_returns['daily_return'] = df_returns[price_col].pct_change()
        
        # Анализируем сигналы
        signal_counts = df_returns[signals_col].value_counts()
        self.logger.debug(f"Распределение сигналов: {signal_counts.to_dict()}")
        
        # Стратегическая доходность с учетом коротких позиций
        # Отрицательный сигнал (-1) означает короткую позицию, инвертируем доходность
        df_returns['strategy_return'] = df_returns[signals_col].shift(1) * df_returns['daily_return']
        
        return df_returns
    
    def calculate_returns_by_ticker(self):
        """
        Рассчитывает доходности для каждого тикера отдельно
        """
        if self.df is None:
            self.logger.error("Данные не загружены")
            return None
            
        self.logger.info("Расчет доходностей по тикерам")
        
        # Анализ сигналов в данных
        if 'signal' in self.df.columns:
            signal_counts = self.df['signal'].value_counts()
            self.logger.info(f"Анализ сигналов в данных: {signal_counts.to_dict()}")
        
        # Расчет доходности для каждого тикера
        if 'ticker' in self.df.columns:
            self.returns_by_ticker = {}
            
            for ticker, ticker_data in self.df.groupby('ticker'):
                ticker_data_sorted = ticker_data.sort_index()
                
                # Анализ сигналов для тикера
                if 'signal' in ticker_data_sorted.columns:
                    signal_counts = ticker_data_sorted['signal'].value_counts()
                    self.logger.info(f"Сигналы для {ticker}: {signal_counts.to_dict()}")
                
                returns_df = self.calculate_returns(ticker_data_sorted)
                if returns_df is not None:
                    self.returns_by_ticker[ticker] = returns_df
                
            self.logger.info(f"Рассчитаны доходности для {len(self.returns_by_ticker)} тикеров")
        else:
            # Если нет колонки ticker, считаем как для одного актива
            returns_df = self.calculate_returns(self.df)
            if returns_df is not None:
                self.returns_by_ticker = {'SINGLE': returns_df}
                self.logger.info("Рассчитаны доходности для одного актива")
            
        return self.returns_by_ticker
    
    def calculate_bonds_returns(self):
        """
        Рассчитывает доходности для облигаций
        """
        if not self.bonds_details:
            self.logger.info("Нет данных о портфеле облигаций")
            return {}
        
        self.logger.info("Расчет доходностей по облигациям")
        
        # Создаем пустой DataFrame с датами из акций для заполнения
        dates = []
        if self.returns_by_ticker:
            # Берем даты из первого тикера акций
            first_ticker = next(iter(self.returns_by_ticker.values()))
            dates = first_ticker.index
        elif self.df is not None:
            # Или из основных данных
            dates = self.df.index
        
        if len(dates) == 0:
            self.logger.error("Нет дат для расчета доходностей облигаций")
            return {}
        
        # Для каждой облигации создаем DataFrame с доходностью
        for bond_name, details in self.bonds_details.items():
            security_code = details['security_code']
            
            # Если есть исторические данные
            if security_code in self.bonds_df:
                bond_data = self.bonds_df[security_code]
                
                # Расчет доходности на основе исторических данных
                if 'price' in bond_data.columns:
                    bond_data['daily_return'] = bond_data['price'].pct_change()
                    self.bonds_returns[security_code] = bond_data
                elif 'yield' in bond_data.columns:
                    # Преобразуем годовую доходность в дневную
                    annual_yield = bond_data['yield'] / 100
                    bond_data['daily_return'] = (1 + annual_yield) ** (1/252) - 1
                    self.bonds_returns[security_code] = bond_data
                else:
                    self.logger.warning(f"Не найдены данные для расчета доходности облигации {security_code}")
            else:
                # Создаем DataFrame с фиксированной доходностью
                bond_yield = details.get('yield', 0) / 100 # Переводим процент в десятичную дробь
                daily_yield = (1 + bond_yield) ** (1/252) - 1
                
                # Создаем DataFrame с дневной доходностью
                bond_df = pd.DataFrame(index=dates)
                bond_df['daily_return'] = daily_yield
                self.bonds_returns[security_code] = bond_df
                
                self.logger.info(f"Создана фиксированная доходность для облигации {security_code}: {daily_yield*100:.6f}% (дневная)")
        
        self.logger.info(f"Рассчитаны доходности для {len(self.bonds_returns)} облигаций")
        return self.bonds_returns
    
    def calculate_portfolio_return(self, weights=None):
        """
        Рассчитывает доходность портфеля с заданными весами,
        включая детализированную безрисковую часть (облигации) и поддержку коротких позиций
        """
        if weights is None:
            weights = self.portfolio_weights
            
        # Если веса не предоставлены, используем равные веса для акций
        if weights is None:
            if not self.returns_by_ticker:
                self.logger.error("Нет данных по доходностям акций и не предоставлены веса")
                return None
            
            weights = {ticker: 1/len(self.returns_by_ticker) for ticker in self.returns_by_ticker.keys()}
            self.logger.info("Используются равные веса для всех активов")
        
        self.logger.info("Расчет доходности портфеля")
        
        # Анализ структуры весов портфеля
        stock_weights = {k: v for k, v in weights.items() if k != 'RISK_FREE' and k != 'rf_details'}
        self.logger.info(f"Анализ структуры портфельных весов:")
        self.logger.info(f"Веса акций: {stock_weights}")
        
        # Проверка соответствия тикеров
        available_tickers = set(self.returns_by_ticker.keys())
        weight_tickers = set(stock_weights.keys())
        matching_tickers = available_tickers.intersection(weight_tickers)
        missing_in_weights = available_tickers - weight_tickers
        missing_in_data = weight_tickers - available_tickers
        
        self.logger.info(f"Доступно тикеров в данных: {len(available_tickers)}")
        self.logger.info(f"Тикеров в весах портфеля: {len(weight_tickers)}")
        self.logger.info(f"Совпадающих тикеров: {len(matching_tickers)}")
        
        if missing_in_weights:
            self.logger.warning(f"Тикеры в данных, отсутствующие в весах: {missing_in_weights}")
        if missing_in_data:
            self.logger.warning(f"Тикеры в весах, отсутствующие в данных: {missing_in_data}")
        
        # Определяем общие даты для портфеля
        all_dates = set()
        
        # Добавляем даты из акций
        for ticker, returns_df in self.returns_by_ticker.items():
            all_dates.update(returns_df.index)
        
        # Добавляем даты из облигаций
        for security_code, returns_df in self.bonds_returns.items():
            all_dates.update(returns_df.index)
        
        all_dates = sorted(all_dates)
        
        # Создаем DataFrame для портфеля
        portfolio_df = pd.DataFrame(index=all_dates)
        
        # Анализ сигналов для отладки
        signal_stats = {}
        for ticker, returns_df in self.returns_by_ticker.items():
            if 'signal' in returns_df.columns:
                non_zero_signals = (returns_df['signal'] != 0).sum()
                signal_stats[ticker] = {
                    'non_zero_signals': non_zero_signals,
                    'total_rows': len(returns_df),
                    'percent': non_zero_signals / len(returns_df) * 100 if len(returns_df) > 0 else 0
                }
        
        self.logger.info(f"Статистика сигналов в данных:")
        for ticker, stats in signal_stats.items():
            self.logger.info(f"  {ticker}: {stats['non_zero_signals']}/{stats['total_rows']} ненулевых сигналов ({stats['percent']:.1f}%)")
        
        # Добавляем доходности акций с учетом коротких позиций
        stocks_added = 0
        for ticker, returns_df in self.returns_by_ticker.items():
            # Проверяем, есть ли тикер в весах и имеет ли он ненулевой вес
            if ticker in stock_weights and abs(stock_weights[ticker]) > 0.001:
                # Проверяем наличие колонки strategy_return
                if 'strategy_return' not in returns_df.columns:
                    self.logger.warning(f"Колонка 'strategy_return' отсутствует для {ticker}, пропускаем")
                    continue
                    
                # Анализируем значения strategy_return
                non_zero_returns = (returns_df['strategy_return'] != 0).sum()
                self.logger.info(f"Тикер {ticker}: {non_zero_returns}/{len(returns_df)} ненулевых strategy_return")
                    
                weight = stock_weights[ticker]
                ticker_returns = returns_df['strategy_return'].reindex(all_dates)
                
                # Отладочная информация о первых нескольких значениях
                if len(ticker_returns) > 0:
                    first_values = ticker_returns.head(5).tolist()
                    self.logger.info(f"Первые 5 значений strategy_return для {ticker}: {first_values}")
                    
                # Если вес отрицательный (короткая позиция), инвертируем доходность
                if weight < 0:
                    # Вес берем по модулю, доходность с обратным знаком
                    portfolio_df[f'{ticker}_return'] = -ticker_returns * abs(weight)
                    self.logger.info(f"Добавлен тикер {ticker} с КОРОТКОЙ позицией, вес: {weight:.4f}")
                else:
                    portfolio_df[f'{ticker}_return'] = ticker_returns * weight
                    self.logger.info(f"Добавлен тикер {ticker} с ДЛИННОЙ позицией, вес: {weight:.4f}")
                
                stocks_added += 1
        
        self.logger.info(f"Всего добавлено {stocks_added} акций в портфель")
        
        # Извлекаем детали безрисковой части
        rf_details = weights.get('rf_details', {})
        rf_allocation = weights.get('RISK_FREE', 0)
        
        if rf_details:
            self.logger.info(f"Найдена безрисковая часть с весом {rf_allocation:.4f} и {len(rf_details)} облигациями")
        elif 'RISK_FREE' in weights:
            self.logger.info(f"Найдена безрисковая часть с весом {rf_allocation:.4f} без детализации")
        
        # Если есть детали по облигациям, добавляем их
        bonds_added = 0
        if rf_details and rf_allocation > 0:
            for bond_name, details in rf_details.items():
                security_code = details['security_code']
                
                # Вес облигации в общем портфеле = вес в безрисковой части * общий вес безрисковой части
                weight_in_portfolio = details['weight']
                
                if security_code in self.bonds_returns:
                    bond_returns = self.bonds_returns[security_code]['daily_return'].reindex(all_dates)
                    portfolio_df[f'{security_code}_return'] = bond_returns * weight_in_portfolio
                    bonds_added += 1
                    self.logger.info(f"Добавлена облигация {security_code} ({bond_name}) с весом {weight_in_portfolio:.4f}")
                    
                    # Отладочная информация о первых нескольких значениях
                    if len(bond_returns) > 0:
                        first_values = bond_returns.head(5).tolist()
                        self.logger.info(f"Первые 5 значений доходности для облигации {security_code}: {first_values}")
                else:
                    # Если нет данных по этой облигации, используем фиксированную доходность
                    bond_yield = details.get('yield', 0) / 100
                    daily_yield = (1 + bond_yield) ** (1/252) - 1
                    portfolio_df[f'{security_code}_return'] = daily_yield * weight_in_portfolio
                    bonds_added += 1
                    self.logger.info(f"Добавлена облигация {security_code} с фиксированной доходностью {daily_yield*100:.6f}% и весом {weight_in_portfolio:.4f}")
        # Если есть безрисковая часть, но нет деталей, используем общую безрисковую ставку
        elif 'RISK_FREE' in weights and weights['RISK_FREE'] > 0:
            rf_weight = weights['RISK_FREE']
            # Предполагаем безрисковую ставку в годовом исчислении, переводим в дневную
            risk_free_rate = 0.075  # 7.5% годовых
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
            
            portfolio_df['RISK_FREE_return'] = daily_rf_rate * rf_weight
            bonds_added += 1
            self.logger.info(f"Добавлена безрисковая часть с весом {rf_weight:.4f} и ставкой {risk_free_rate*100:.2f}% годовых")
        
        self.logger.info(f"Всего добавлено {bonds_added} облигаций в портфель")
        
        # Проверяем, что в portfolio_df есть хотя бы одна колонка с доходностями
        if portfolio_df.empty or portfolio_df.shape[1] == 0:
            self.logger.error("Не удалось добавить ни одного актива в портфель!")
            # Создаем пустой портфель с нулевыми доходностями для предотвращения ошибок
            portfolio_df['portfolio_return'] = 0.0
        else:
            # Рассчитываем суммарную доходность портфеля
            portfolio_df['portfolio_return'] = portfolio_df.sum(axis=1)
            
            # Отладочная информация о портфельной доходности
            non_zero_portfolio = (portfolio_df['portfolio_return'] != 0).sum()
            self.logger.info(f"Портфельная доходность: {non_zero_portfolio}/{len(portfolio_df)} ненулевых значений")
            
            if len(portfolio_df) > 0:
                first_values = portfolio_df['portfolio_return'].head(5).tolist()
                self.logger.info(f"Первые 5 значений portfolio_return: {first_values}")
        
        # Заполняем пропуски нулями
        portfolio_df = portfolio_df.fillna(0)
        
        self.portfolio_returns = portfolio_df
        
        self.logger.info(f"Рассчитана доходность портфеля за {len(portfolio_df)} дней")
        return portfolio_df
    
    def calculate_performance_metrics(self, returns=None, risk_free_rate=0.075):
        """
        Рассчитывает метрики эффективности стратегии
        """
        if returns is None:
            returns = self.portfolio_returns
            
        if returns is None or 'portfolio_return' not in returns.columns:
            self.logger.error("Доходности портфеля не рассчитаны")
            return None
            
        self.logger.info("Расчет метрик эффективности")
        
        daily_returns = returns['portfolio_return'].dropna()
        
        if len(daily_returns) == 0:
            self.logger.error("Нет данных для расчета метрик")
            return None
        
        # Кумулятивная доходность
        cumulative_return = (1 + daily_returns).cumprod().iloc[-1] - 1
        
        # Годовая доходность
        annual_return = (1 + cumulative_return) ** (252 / len(daily_returns)) - 1
        
        # Годовая волатильность
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Коэффициент Шарпа
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Максимальная просадка
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Винрейт (доля доходных дней)
        win_rate = (daily_returns > 0).mean()
        
        # Сортино (учет только отрицательной волатильности)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Калмар (отношение доходности к максимальной просадке)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Подсчет просадок
        drawdown_series = drawdown.copy()
        drawdown_threshold = -0.05  # 5% просадка
        
        in_drawdown = False
        drawdown_count = 0
        drawdown_days = 0
        drawdown_days_list = []
        
        for dd in drawdown_series:
            if dd <= drawdown_threshold and not in_drawdown:
                in_drawdown = True
                drawdown_count += 1
                drawdown_days = 1
            elif dd <= drawdown_threshold and in_drawdown:
                drawdown_days += 1
            elif dd > drawdown_threshold and in_drawdown:
                in_drawdown = False
                drawdown_days_list.append(drawdown_days)
                drawdown_days = 0
        
        # Если последняя просадка не закончилась, добавляем ее
        if in_drawdown:
            drawdown_days_list.append(drawdown_days)
        
        avg_drawdown_days = np.mean(drawdown_days_list) if drawdown_days_list else 0
        
        metrics = {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'drawdown_count': drawdown_count,
            'avg_drawdown_days': avg_drawdown_days
        }
        
        self.metrics = metrics
        
        self.logger.info(f"Кумулятивная доходность: {cumulative_return*100:.2f}%, " + 
                         f"Годовая доходность: {annual_return*100:.2f}%, " + 
                         f"Шарп: {sharpe_ratio:.2f}, " + 
                         f"Макс. просадка: {max_drawdown*100:.2f}%")
        
        return metrics
    
    def visualize_results(self, output_dir=None):
        """
        Визуализирует результаты бэктеста с поддержкой коротких позиций
        """
        if self.portfolio_returns is None or self.metrics is None:
            self.logger.error("Результаты бэктеста не рассчитаны")
            return None
                
        if output_dir:
            # Используем новую директорию для улучшенных визуализаций
            output_dir = output_dir.replace('signal_visualizations', 'signal_visualizations_improved')
            os.makedirs(output_dir, exist_ok=True)
                
        self.logger.info("Создание визуализации результатов бэктеста")
        
        # 1. График кумулятивной доходности
        cumulative_returns = (1 + self.portfolio_returns['portfolio_return'].fillna(0)).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label='Стратегия')
        
        # Если есть другие метрики для сравнения, добавляем их
        if 'daily_return' in self.portfolio_returns.columns:
            benchmark_returns = (1 + self.portfolio_returns['daily_return'].fillna(0)).cumprod()
            plt.plot(benchmark_returns.index, benchmark_returns, linestyle='--', label='Бенчмарк')
        
        plt.title('Кумулятивная доходность')
        plt.xlabel('Дата')
        plt.ylabel('Доходность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'cumulative_return.png'))
            self.logger.info(f"График кумулятивной доходности сохранен в {output_dir}/cumulative_return.png")
        else:
            plt.show()
            
        plt.close()
        
        # 2. График просадок
        daily_returns = self.portfolio_returns['portfolio_return'].dropna()
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown.index, drawdown * 100)
        plt.title('Просадки портфеля (%)')
        plt.xlabel('Дата')
        plt.ylabel('Просадка (%)')
        plt.grid(True, alpha=0.3)
        plt.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'drawdown.png'))
            self.logger.info(f"График просадок сохранен в {output_dir}/drawdown.png")
        else:
            plt.show()
            
        plt.close()
        
        # 3. График распределения доходностей (гистограмма)
        plt.figure(figsize=(12, 6))
        daily_returns.hist(bins=50, alpha=0.7, density=True)
        
        # Добавляем нормальное распределение для сравнения
        import scipy.stats as stats
        x = np.linspace(daily_returns.min(), daily_returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, daily_returns.mean(), daily_returns.std()), 
                'r--', linewidth=2, label='Нормальное распределение')
        
        plt.title('Распределение дневных доходностей')
        plt.xlabel('Дневная доходность')
        plt.ylabel('Плотность')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'returns_distribution.png'))
            self.logger.info(f"График распределения доходностей сохранен в {output_dir}/returns_distribution.png")
        else:
            plt.show()
        
        plt.close()
        
        # 4. График структуры портфеля с учетом коротких позиций
        try:
            # Получаем веса из портфеля
            if hasattr(self, 'portfolio_weights') and self.portfolio_weights:
                weights = self.portfolio_weights.copy()  # Используем копию для безопасности
            else:
                # Если веса не предоставлены, считаем их из вклада в доходность
                asset_weights = {}
                for col in self.portfolio_returns.columns:
                    if col != 'portfolio_return' and col.endswith('_return'):
                        ticker = col.replace('_return', '')
                        if self.portfolio_returns['portfolio_return'].mean() != 0:
                            asset_weights[ticker] = self.portfolio_returns[col].mean() / self.portfolio_returns['portfolio_return'].mean()
                        else:
                            asset_weights[ticker] = 0
                weights = asset_weights
            
            # Разделяем на длинные и короткие позиции
            long_weights = {}
            short_weights = {}
            other_weights = {}
            
            for ticker, weight in weights.items():
                # Пропускаем нечисловые и специальные ключи
                if not isinstance(weight, (int, float)) or ticker == 'rf_details':
                    continue
                    
                # Обрабатываем риск-фри отдельно
                if ticker == 'RISK_FREE':
                    if weight > 0:  # Только положительное значение для риск-фри
                        other_weights[ticker] = weight
                # Явно помеченные SHORT/LONG в названии тикера
                elif '_SHORT' in ticker:
                    short_ticker = ticker.replace('_SHORT', '')
                    short_weights[short_ticker] = abs(weight)  # Берем модуль для отображения
                elif '_LONG' in ticker:
                    long_ticker = ticker.replace('_LONG', '')
                    if weight > 0:  # Гарантируем положительность
                        long_weights[long_ticker] = weight
                # По знаку значения: отрицательные - SHORT, положительные - LONG
                elif weight < 0:
                    short_weights[ticker] = abs(weight)  # Берем модуль для отображения
                elif weight > 0:
                    long_weights[ticker] = weight
            
            # Создаем два графика: один для длинных, другой для коротких позиций
            plt.figure(figsize=(18, 10))
            
            # Проверяем наличие длинных и коротких позиций
            has_long = bool(long_weights)
            has_short = bool(short_weights)
            
            # Определяем размещение графиков
            if has_long and has_short:
                # Оба типа позиций
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
            elif has_long:
                # Только длинные
                ax1 = plt.subplot(1, 1, 1)
                ax2 = None
            elif has_short:
                # Только короткие
                ax1 = None
                ax2 = plt.subplot(1, 1, 1)
            else:
                # Нет значимых позиций
                ax1 = plt.subplot(1, 1, 1)
                ax2 = None
                ax1.text(0.5, 0.5, "Нет значимых позиций в портфеле", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=14)
                
            # 1. Длинные позиции
            if has_long and ax1 is not None:
                # Фильтруем только значимые веса (и гарантируем положительность)
                significant_longs = {k: v for k, v in long_weights.items() if v > 0.01}
                if 'RISK_FREE' in other_weights and other_weights['RISK_FREE'] > 0.01:
                    significant_longs['RISK_FREE'] = other_weights['RISK_FREE']
                    
                # Группируем малые веса, если нужно
                other_long = sum(v for k, v in long_weights.items() if v <= 0.01 and v > 0)
                if other_long > 0.01:
                    significant_longs['Другие LONG'] = other_long
                    
                if significant_longs:
                    # Дополнительно проверяем, что все значения положительные
                    if all(v > 0 for v in significant_longs.values()):
                        ax1.pie(
                            list(significant_longs.values()),
                            labels=list(significant_longs.keys()),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=plt.cm.Greens(np.linspace(0.5, 0.8, len(significant_longs)))
                        )
                        ax1.set_title('Длинные позиции (LONG)')
                        ax1.axis('equal')
                    else:
                        ax1.text(0.5, 0.5, "Ошибка: обнаружены отрицательные значения в длинных позициях", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax1.transAxes, fontsize=12, color='red')
                else:
                    ax1.text(0.5, 0.5, "Нет значимых длинных позиций", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax1.transAxes, fontsize=14)
            
            # 2. Короткие позиции
            if has_short and ax2 is not None:
                # Фильтруем только значимые веса (все уже положительные, т.к. мы взяли abs)
                significant_shorts = {k: v for k, v in short_weights.items() if v > 0.01}
                    
                # Группируем малые веса, если нужно
                other_short = sum(v for k, v in short_weights.items() if v <= 0.01 and v > 0)
                if other_short > 0.01:
                    significant_shorts['Другие SHORT'] = other_short
                    
                if significant_shorts:
                    # Дополнительно проверяем, что все значения положительные
                    if all(v > 0 for v in significant_shorts.values()):
                        ax2.pie(
                            list(significant_shorts.values()),
                            labels=list(significant_shorts.keys()),
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=plt.cm.Reds(np.linspace(0.5, 0.8, len(significant_shorts)))
                        )
                        ax2.set_title('Короткие позиции (SHORT)')
                        ax2.axis('equal')
                    else:
                        ax2.text(0.5, 0.5, "Ошибка: обнаружены отрицательные значения в коротких позициях", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax2.transAxes, fontsize=12, color='red')
                else:
                    ax2.text(0.5, 0.5, "Нет значимых коротких позиций", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=14)
            
            plt.suptitle('Структура портфеля', fontsize=16)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'portfolio_structure.png'))
                self.logger.info(f"График структуры портфеля сохранен в {output_dir}/portfolio_structure.png")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Ошибка при создании графика структуры портфеля: {e}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Не удалось создать график структуры портфеля: {e}", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'portfolio_structure_error.png'))
            else:
                plt.show()
                
        plt.close()
        
        # 5. Дополнительный график - доходность по месяцам
        try:
            # Попытка создать календарь доходностей по месяцам
            monthly_returns = self.portfolio_returns['portfolio_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Создаем сводную таблицу для календаря доходностей
            returns_by_year_month = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })
            
            pivot_table = returns_by_year_month.pivot(index='year', columns='month', values='return')
            
            # Преобразуем числовые месяцы в названия
            month_names = {
                1: 'Янв', 2: 'Фев', 3: 'Мар', 4: 'Апр', 5: 'Май', 6: 'Июн', 
                7: 'Июл', 8: 'Авг', 9: 'Сен', 10: 'Окт', 11: 'Ноя', 12: 'Дек'
            }
            pivot_table.columns = [month_names[m] for m in pivot_table.columns]
            
            # Создаем тепловую карту
            plt.figure(figsize=(12, 8))
            heatmap = plt.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
            plt.colorbar(heatmap, label='Доходность')
            
            # Подписи
            plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=0)
            plt.yticks(range(len(pivot_table.index)), pivot_table.index)
            
            # Добавляем значения в ячейки
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    if not np.isnan(pivot_table.values[i, j]):
                        plt.text(j, i, f"{pivot_table.values[i, j]*100:.1f}%", 
                                ha="center", va="center", color="black")
            
            plt.title('Календарь месячных доходностей')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'monthly_returns_calendar.png'))
                self.logger.info(f"Календарь месячных доходностей сохранен в {output_dir}/monthly_returns_calendar.png")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            self.logger.warning(f"Не удалось создать календарь доходностей: {e}")
        
        # Сохранение метрик в файл
        if output_dir:
            with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
                f.write(f"Период: {self.portfolio_returns.index.min().strftime('%Y-%m-%d')} - {self.portfolio_returns.index.max().strftime('%Y-%m-%d')}\n\n")
                f.write(f"Кумулятивная доходность: {self.metrics['cumulative_return']*100:.2f}%\n")
                f.write(f"Годовая доходность: {self.metrics['annual_return']*100:.2f}%\n")
                f.write(f"Годовая волатильность: {self.metrics['annual_volatility']*100:.2f}%\n")
                f.write(f"Коэффициент Шарпа: {self.metrics['sharpe_ratio']:.2f}\n")
                f.write(f"Коэффициент Сортино: {self.metrics.get('sortino_ratio', 0):.2f}\n")
                f.write(f"Коэффициент Калмар: {self.metrics.get('calmar_ratio', 0):.2f}\n")
                f.write(f"Максимальная просадка: {self.metrics['max_drawdown']*100:.2f}%\n")
                f.write(f"Винрейт: {self.metrics['win_rate']*100:.2f}%\n\n")
                
                # Добавляем информацию о структуре портфеля
                f.write("СТРУКТУРА ПОРТФЕЛЯ:\n")
                f.write("------------------\n")
                
                if long_weights:
                    f.write("\nДлинные позиции (LONG):\n")
                    for ticker, weight in sorted(long_weights.items(), key=lambda x: x[1], reverse=True):
                        if weight > 0.01:
                            f.write(f"  {ticker}: {weight*100:.2f}%\n")
                        
                if short_weights:
                    f.write("\nКороткие позиции (SHORT):\n")
                    for ticker, weight in sorted(short_weights.items(), key=lambda x: x[1], reverse=True):
                        if weight > 0.01:
                            f.write(f"  {ticker}: -{weight*100:.2f}%\n")
                            
                if 'RISK_FREE' in other_weights:
                    f.write(f"\nБезрисковые активы: {other_weights['RISK_FREE']*100:.2f}%\n")
            
            # Сохранение портфельных доходностей
            self.portfolio_returns.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))
            
            self.logger.info(f"Метрики и данные сохранены в {output_dir}")

    def run_pipeline(self, input_file=None, portfolio_weights=None, 
               start_date=None, end_date=None, output_dir=f"{BASE_PATH}/data/portfolio/results", 
               risk_free_rate=0.075, bonds_data_dir=None):
        """
        Запускает полный пайплайн бэктестирования

        Parameters
        -----------
        input_file : str, optional
            Путь к файлу с данными и сигналами
        portfolio_weights : dict, optional
            Словарь с весами активов в портфеле
        start_date, end_date : str, optional
            Даты начала и конца бэктеста в формате 'YYYY-MM-DD'
        output_dir : str, optional
            Директория для сохранения результатов
        risk_free_rate : float
            Безрисковая ставка для расчета метрик
        bonds_data_dir : str, optional
            Директория с историческими данными по облигациям

        Returns
        --------
        dict с результатами бэктеста
        """
        
        self.logger.info("Запуск пайплайна бэктестирования")

        # Обновление параметров, если указаны
        if input_file is not None:
            self.input_file = input_file
        if portfolio_weights is not None:
            self.portfolio_weights = portfolio_weights
        if start_date is not None:
            self.start_date = pd.to_datetime(start_date)
        if end_date is not None:
            self.end_date = pd.to_datetime(end_date)
        if bonds_data_dir is not None:
            self.bonds_data_dir = bonds_data_dir
            
        # Шаг 1: Загрузка данных по акциям
        self.load_data()
        
        if self.df is None or len(self.df) == 0:
            self.logger.error("Не удалось загрузить данные по акциям для бэктеста")
            return None
        
        # Шаг 2: Загрузка данных по облигациям
        if self.bonds_details:
            self.load_bonds_data()
            
        # Шаг 3: Расчет доходностей по тикерам акций
        self.calculate_returns_by_ticker()
        
        if not self.returns_by_ticker:
            self.logger.error("Не удалось рассчитать доходности по тикерам акций")
            return None
            
        # Шаг 4: Расчет доходностей по облигациям
        if self.bonds_details:
            self.calculate_bonds_returns()
            
        # Шаг 5: Расчет доходности портфеля
        self.calculate_portfolio_return()
        
        if self.portfolio_returns is None:
            self.logger.error("Не удалось рассчитать доходность портфеля")
            return None
            
        # Шаг 6: Расчет метрик эффективности
        self.calculate_performance_metrics(risk_free_rate=risk_free_rate)
        
        if self.metrics is None:
            self.logger.error("Не удалось рассчитать метрики эффективности")
            return None
            
        # Шаг 7: Визуализация результатов
        if output_dir:
            self.visualize_results(output_dir)
            
        # Возвращаем результаты
        results = {
            'metrics': self.metrics,
            'returns': self.portfolio_returns,
            'cumulative_returns': (1 + self.portfolio_returns['portfolio_return'].fillna(0)).cumprod()
        }
        
        return results
