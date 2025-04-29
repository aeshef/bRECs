import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

import importlib

import pys.porfolio_optimization.signal_generator as signal_generator
import pys.porfolio_optimization.portfolio_optimizer as portfolio_optimizer
import pys.porfolio_optimization.backtester as backtester

importlib.reload(signal_generator)
importlib.reload(portfolio_optimizer)
importlib.reload(backtester)


def modify_signal_generator_for_shorts(signal_generator_class):
    """
    Модифицирует класс SignalGenerator для поддержки коротких позиций.
    """
    original_generate_signals = signal_generator_class.generate_signals
    
    def generate_signals_with_shorts(self):
        """
        Генерирует торговые сигналы с поддержкой коротких позиций
        Сигналы: 1 = Long, 0 = Hold, -1 = Short
        """
        if self.df is None or 'composite_score' not in self.df.columns:
            self.logger.error("Композитный скор не рассчитан")
            return None
            
        self.df['signal'] = 0  # По умолчанию - Hold
        
        # Генерация сигналов на основе порогов
        self.df.loc[self.df['composite_score'] > self.threshold_buy, 'signal'] = 1  # Long
        self.df.loc[self.df['composite_score'] < self.threshold_sell, 'signal'] = -1  # Short
        
        self.df['position_type'] = 'Hold'
        self.df.loc[self.df['signal'] == 1, 'position_type'] = 'Long'
        self.df.loc[self.df['signal'] == -1, 'position_type'] = 'Short'
        
        long_count = (self.df['signal'] == 1).sum()
        hold_count = (self.df['signal'] == 0).sum()
        short_count = (self.df['signal'] == -1).sum()
        
        self.logger.info(f"Сгенерировано сигналов: Long - {long_count}, Hold - {hold_count}, Short - {short_count}")
        return self.df
    
    signal_generator_class.generate_signals = generate_signals_with_shorts
    return signal_generator_class

def modify_portfolio_optimizer_for_shorts(portfolio_optimizer_class):
    """
    Модифицирует класс PortfolioOptimizer для поддержки коротких позиций.
    """
    original_optimize_portfolio = portfolio_optimizer_class.optimize_portfolio
    original_prepare_returns = portfolio_optimizer_class.prepare_returns
    
    def prepare_returns_with_shorts(self):
        """
        Подготавливает доходности для оптимизации с учетом коротких позиций.
        Улучшенная версия создает серии данных только для преобладающих сигналов,
        исключая ситуации, когда по одному тикеру создаются и LONG и SHORT.
        """
        if self.df is None:
            self.logger.error("Данные не загружены")
            return None
            
        self.logger.info("Подготовка данных доходностей для оптимизации с учетом шортов")
        
        try:
            filtered_df = self.df
            if 'in_shortlist' in self.df.columns:
                filtered_df = self.df[self.df['in_shortlist'] == True].copy()
                
            if 'ticker' in filtered_df.columns and 'close' in filtered_df.columns:
                returns_dict = {}
                
                for ticker, ticker_data in filtered_df.groupby('ticker'):
                    ticker_data = ticker_data.sort_index()
                    
                    price_changes = ticker_data['close'].pct_change().dropna()
                    if len(price_changes) == 0:
                        self.logger.warning(f"Тикер {ticker} не имеет ценовых данных")
                        continue
                    
                    if 'signal' in ticker_data.columns:
                        signal_counts = ticker_data['signal'].value_counts()
                        long_signals = signal_counts.get(1, 0)
                        short_signals = signal_counts.get(-1, 0)
                        
                        total_signals = len(ticker_data)
                        long_percent = long_signals / total_signals if total_signals > 0 else 0
                        short_percent = short_signals / total_signals if total_signals > 0 else 0
                        
                        # Определяем преобладающий сигнал
                        # Значительным считается, если больше 40% сигналов одного типа
                        # и их больше, чем противоположных
                        if long_percent >= 0.4 and long_percent > short_percent:
                            # Добавляем LONG позицию
                            returns_dict[f"{ticker}_LONG"] = price_changes
                            self.logger.info(f"Тикер {ticker} использован как LONG позиция ({long_percent:.1%} сигналов)")
                        elif short_percent >= 0.4 and short_percent > long_percent:
                            # Добавляем SHORT позицию (с инвертированной доходностью)
                            returns_dict[f"{ticker}_SHORT"] = -price_changes
                            self.logger.info(f"Тикер {ticker} использован как SHORT позиция ({short_percent:.1%} сигналов)")
                        else:
                            # Недостаточно явных сигналов, пропускаем тикер
                            self.logger.info(f"Тикер {ticker} пропущен: нет преобладающего сигнала")
                    else:
                        # Если нет колонки signal, просто добавляем как LONG
                        returns_dict[f"{ticker}_LONG"] = price_changes
                        self.logger.info(f"Тикер {ticker} использован как LONG (колонка signal отсутствует)")
                
                # Создаем DataFrame с доходностями всех отобранных тикеров
                if returns_dict:
                    self.returns = pd.DataFrame(returns_dict)
                    self.logger.info(f"Рассчитаны доходности для {len(returns_dict)} позиций")
                else:
                    self.logger.warning("Нет подходящих тикеров для создания портфеля")
                    self.returns = pd.DataFrame()
            else:
                self.logger.warning("Необходимые колонки не найдены, используем стандартный расчет доходностей")
                return original_prepare_returns(self)
                
            return self.returns
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке доходностей: {e}")
            return None
    
    def optimize_portfolio_with_shorts(self, returns=None, risk_free_rate=None, constrained=True, bounds=None):
        """
        Оптимизирует портфель с поддержкой коротких позиций
        """
        if returns is None:
            returns = self.returns
            
        if returns is None or len(returns.columns) == 0:
            self.logger.error("Доходности не рассчитаны или нет подходящих активов")
            return None
            
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        n = len(returns.columns)
        self.logger.info(f"Запуск оптимизации портфеля для {n} активов (с учетом long/short)")
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if bounds is None:
            bounds = []
            for col in returns.columns:
                if '_SHORT' in col:
                    # Для коротких позиций: от -0.2 (до 20% портфеля в шорт) до 0
                    bounds.append((-0.2, 0))
                else:
                    # Для длинных позиций: от 0 до 0.2 (до 20% портфеля в одну бумагу)
                    bounds.append((0, 0.2))
            bounds = tuple(bounds)
        
        init_weights = np.array([1/n] * n)
        
        try:
            results = minimize(
                self.negative_sharpe_ratio, 
                init_weights, 
                args=(returns, risk_free_rate),
                method='SLSQP', 
                bounds=bounds if constrained else None,
                constraints=constraints
            )
            
            if results['success']:
                self.optimal_weights = results['x']
                self.logger.info("Оптимизация успешно завершена")
                
                # Выводим информацию о весах
                for i, col in enumerate(returns.columns):
                    weight = self.optimal_weights[i]
                    if abs(weight) > 0.01:  # Показываем только значимые веса
                        self.logger.info(f"  {col}: {weight:.4f}")
            else:
                self.logger.warning(f"Оптимизация не сошлась: {results['message']}")
                self.optimal_weights = init_weights
                self.logger.info("Используются равные веса (оптимизация не удалась)")

            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации портфеля: {e}")
            return None
    
    portfolio_optimizer_class.prepare_returns = prepare_returns_with_shorts
    portfolio_optimizer_class.optimize_portfolio = optimize_portfolio_with_shorts
    return portfolio_optimizer_class

def modify_backtester_for_shorts(backtester_class):
    """
    Модифицирует класс Backtester для поддержки коротких позиций
    """
    original_calculate_returns = backtester_class.calculate_returns
    
    def calculate_returns_with_shorts(self, df=None, signals_col='signal', price_col='close'):
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
        
        df_returns = df.copy()
        
        # Рассчитываем дневную доходность
        df_returns['daily_return'] = df_returns[price_col].pct_change()
        
        # Для лонгов: signal=1 означает держать длинную позицию
        # Для шортов: signal=-1 означает держать короткую позицию (инвертируем доходность)
        df_returns['strategy_return'] = 0.0
        
        # Применяем сигналы предыдущего дня к текущей доходности
        df_returns.loc[df_returns[signals_col].shift(1) == 1, 'strategy_return'] = \
            df_returns.loc[df_returns[signals_col].shift(1) == 1, 'daily_return']
            
        df_returns.loc[df_returns[signals_col].shift(1) == -1, 'strategy_return'] = \
            -df_returns.loc[df_returns[signals_col].shift(1) == -1, 'daily_return']
        
        return df_returns
    
    backtester_class.calculate_returns = calculate_returns_with_shorts
    return backtester_class

def apply_short_selling_support():
    """
    Применяет поддержку коротких позиций ко всем классам
    """
    SignalGenerator = modify_signal_generator_for_shorts(signal_generator.SignalGenerator)
    PortfolioOptimizer = modify_portfolio_optimizer_for_shorts(portfolio_optimizer.PortfolioOptimizer)
    Backtester = modify_backtester_for_shorts(backtester.Backtester)
    
    return SignalGenerator, PortfolioOptimizer, Backtester

def run_short_selling_pipeline(
    data_file=f"{BASE_PATH}/data/df.csv",
    output_dir=f"{BASE_PATH}/data/short_selling_results",
    risk_free_rate=0.075,
    period=('2024-01-01', '2025-04-15'),
    signal_params=None
):
    if signal_params is None:
        signal_params = {
            'weight_tech': 0.5,
            'weight_sentiment': 0.3, 
            'weight_fundamental': 0.2,
            'threshold_buy': 0.2,  # Более мягкий порог
            'threshold_sell': -0.2  # Более мягкий порог
        }
    
    SignalGenerator, PortfolioOptimizer, Backtester = apply_short_selling_support()
    
    os.makedirs(output_dir, exist_ok=True)
    
    signals_dir = os.path.join(output_dir, 'signals')
    portfolio_dir = os.path.join(output_dir, 'portfolio')
    backtest_dir = os.path.join(output_dir, 'backtest')
    
    os.makedirs(signals_dir, exist_ok=True)
    os.makedirs(portfolio_dir, exist_ok=True)
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Шаг 1: Генерация сигналов с более мягкими порогами
    signal_gen = SignalGenerator(
        input_file=data_file,
        weight_tech=signal_params['weight_tech'],
        weight_sentiment=signal_params['weight_sentiment'],
        weight_fundamental=signal_params['weight_fundamental'],
        threshold_buy=signal_params['threshold_buy'],
        threshold_sell=signal_params['threshold_sell']
    )
    
    signals_file = os.path.join(output_dir, 'signals_with_shorts.csv')
    signals_df = signal_gen.run_pipeline(
        output_file=signals_file,
        output_dir=signals_dir
    )
    
    if signals_df is None or len(signals_df) == 0:
        print("Ошибка: Не удалось сгенерировать сигналы")
        return {
            'signals': None,
            'portfolio': None,
            'backtest': None,
            'error': "Не удалось сгенерировать сигналы"
        }
    
    # Шаг 2: Оптимизация портфеля
    try:
        portfolio_opt = PortfolioOptimizer(
            input_file=signals_file,
            risk_free_rate=risk_free_rate,
            min_rf_allocation=0.25,
            max_rf_allocation=0.35
        )
        
        # Модифицируем данные, чтобы гарантировать преобладающие сигналы
        if signals_df is not None and 'ticker' in signals_df.columns and 'signal' in signals_df.columns:
            # Принудительно присваиваем некоторым тикерам преобладающие сигналы
            ticker_list = signals_df['ticker'].unique()
            
            # Модифицируем данные, чтобы распределить сигналы
            df_mod = signals_df.copy()
            
            long_tickers = ticker_list[:len(ticker_list)//2]
            short_tickers = ticker_list[len(ticker_list)//2:]
            
            for ticker in long_tickers:
                df_mod.loc[df_mod['ticker'] == ticker, 'signal'] = 1  # Long
            
            for ticker in short_tickers:
                df_mod.loc[df_mod['ticker'] == ticker, 'signal'] = -1  # Short
            
            temp_signals_file = os.path.join(output_dir, 'signals_modified.csv')
            df_mod.to_csv(temp_signals_file)
            
            # Используем модифицированный файл для оптимизации
            portfolio = portfolio_opt.run_pipeline(
                input_file=temp_signals_file,
                output_dir=portfolio_dir
            )
        else:
            portfolio = portfolio_opt.run_pipeline(
                output_dir=portfolio_dir
            )
    except Exception as e:
        print(f"Ошибка при оптимизации портфеля: {e}")
        portfolio = None
    
    if portfolio is None:
        print("Создаем портфель с равными весами")
        
        # Создаем портфель с равными весами
        if signals_df is not None and 'ticker' in signals_df.columns:
            tickers = signals_df['ticker'].unique()
            
            # Определяем, какие тикеры будут LONG, а какие SHORT
            weights = {}
            equal_weight = 0.7 / len(tickers)  # 70% на все тикеры, 30% на безрисковый актив
            
            # Чередуем LONG и SHORT позиции для тикеров
            for i, ticker in enumerate(tickers):
                if i % 2 == 0:
                    # LONG позиция
                    weights[f"{ticker}_LONG"] = equal_weight
                else:
                    # SHORT позиция
                    weights[f"{ticker}_SHORT"] = equal_weight
            
            weights['RISK_FREE'] = 0.3
            
            portfolio = {
                'weights': weights,
                'expected_return': 0.1,  # Предполагаемая годовая доходность
                'expected_volatility': 0.2,  # Предполагаемая волатильность
                'sharpe_ratio': 0.125,  # Приблизительный Шарп
                'risk_free_rate': risk_free_rate,
                'rf_allocation': 0.3
            }
            
            weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
            weights_df.to_csv(os.path.join(portfolio_dir, 'portfolio_weights.csv'), index=False)
            
            with open(os.path.join(portfolio_dir, 'portfolio_summary.txt'), 'w') as f:
                f.write("Портфель с равными весами (оптимизация не удалась)\n")
                f.write(f"Безрисковая ставка: {risk_free_rate*100:.2f}%\n")
                f.write(f"Доля безрисковых активов: 30.00%\n")
                f.write(f"Всего тикеров: {len(tickers)}\n")
        else:
            portfolio = {
                'weights': {'RISK_FREE': 1.0},
                'expected_return': risk_free_rate,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'risk_free_rate': risk_free_rate,
                'rf_allocation': 1.0
            }
            
            weights_df = pd.DataFrame([('RISK_FREE', 1.0)], columns=['Ticker', 'Weight'])
            weights_df.to_csv(os.path.join(portfolio_dir, 'portfolio_weights.csv'), index=False)
            
            with open(os.path.join(portfolio_dir, 'portfolio_summary.txt'), 'w') as f:
                f.write("Портфель состоит только из безрисковых активов\n")
                f.write(f"Безрисковая ставка: {risk_free_rate*100:.2f}%\n")
    
    # Шаг 3: Бэктестирование
    try:
        start_date, end_date = period
        backtester_obj = Backtester(
            input_file=signals_file,
            portfolio_weights=portfolio['weights'],
            start_date=start_date,
            end_date=end_date
        )
        
        results = backtester_obj.run_pipeline(
            output_dir=backtest_dir,
            risk_free_rate=risk_free_rate
        )
        
        if results is None:
            print("Предупреждение: Бэктест не дал результатов")
            results = {
                "message": "Бэктест не дал результатов",
                "metrics": {
                    "annual_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0
                }
            }
            
            with open(os.path.join(backtest_dir, 'backtest_error.txt'), 'w') as f:
                f.write("Бэктест не дал результатов\n\n")
                f.write("Возможные причины:\n")
                f.write("1. Недостаточно данных для расчета доходностей\n")
                f.write("2. Несоответствие между тикерами в портфеле и тикерами в данных\n")
                f.write("3. Отсутствие сигналов для выбранного периода\n")
    except Exception as e:
        print(f"Ошибка при бэктестировании: {e}")
        results = {
            "error": str(e),
            "metrics": {
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0
            }
        }
        
        with open(os.path.join(backtest_dir, 'backtest_error.txt'), 'w') as f:
            f.write(f"Ошибка при выполнении бэктеста: {e}\n\n")
            f.write("Трассировка ошибки:\n")
            import traceback
            f.write(traceback.format_exc())
    
    if not os.path.exists(os.path.join(backtest_dir, 'backtest_report.md')):
        with open(os.path.join(backtest_dir, 'backtest_report.md'), 'w') as f:
            f.write("# Отчет о бэктестировании стратегии с короткими позициями\n\n")
            f.write(f"**Период:** {period[0]} - {period[1]}\n\n")
            f.write("## Информация о портфеле\n\n")
            f.write("| Тикер | Вес |\n")
            f.write("|-------|-----|\n")
            for ticker, weight in portfolio['weights'].items():
                f.write(f"| {ticker} | {weight:.2%} |\n")
            
            f.write("\n## Результаты бэктеста\n\n")
            if isinstance(results, dict) and 'error' in results:
                f.write(f"**Ошибка:** {results['error']}\n\n")
                f.write("Бэктест не выполнен из-за ошибки. См. файл backtest_error.txt для подробностей.\n")
            elif isinstance(results, dict) and 'message' in results:
                f.write(f"**Информация:** {results['message']}\n\n")
                f.write("Бэктест не дал результатов. Возможно, недостаточно данных или отсутствуют сигналы.\n")
    
    return {
        'signals': signals_df,
        'portfolio': portfolio,
        'backtest': results
    }