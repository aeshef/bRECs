import pandas as pd
import numpy as np
import os
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
import importlib
import sys

sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/porfolio_optimization')

import signal_generator
importlib.reload(signal_generator)
from signal_generator import SignalGenerator, run_pipeline_signal_generator

import portfolio_optimizer
importlib.reload(portfolio_optimizer)
from portfolio_optimizer import PortfolioOptimizer

import backtester
importlib.reload(backtester)
from backtester import Backtester

sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys')

from utils.logger import BaseLogger

class HonestBacktester(BaseLogger):
    """
    Класс для проведения честного бэктеста на будущий период
    """
    
    def __init__(self, 
                data_file="/Users/aeshef/Documents/GitHub/kursach/data/df.csv",
                best_params_file=None,
                train_period=('2024-01-01', '2024-12-31'),
                test_period=('2025-01-01', '2025-06-30'),
                output_dir="/Users/aeshef/Documents/GitHub/kursach/data/honest_backtest",
                risk_free_rate=0.075,
                use_grid_search_params=True):
        """
        Инициализация бэктестера
        
        Parameters:
        -----------
        data_file : str
            Путь к файлу с данными
        best_params_file : str, optional
            Путь к файлу с лучшими параметрами из Grid Search
        train_period : tuple
            Период обучения (начало, конец)
        test_period : tuple
            Период тестирования (начало, конец)
        output_dir : str
            Директория для сохранения результатов
        risk_free_rate : float
            Безрисковая ставка
        use_grid_search_params : bool
            Использовать ли параметры из Grid Search
        """
        super().__init__('HonestBacktester')
        
        self.data_file = data_file
        self.best_params_file = best_params_file
        self.train_period = train_period
        self.test_period = test_period
        self.output_dir = output_dir
        self.risk_free_rate = risk_free_rate
        self.use_grid_search_params = use_grid_search_params
        
        os.makedirs(output_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(output_dir, 'honest_backtest.log'))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("HonestBacktester initialized")
    
    def run(self):
        """
        Запускает честный бэктест на будущий период
        
        Returns:
        --------
        dict с результатами бэктеста
        """
        # Загружаем данные
        self.logger.info(f"Загрузка данных из {self.data_file}")
        try:
            df = pd.read_csv(self.data_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            return None
        
        # Разделение на тренировочные и тестовые данные
        train_start, train_end = self.train_period
        test_start, test_end = self.test_period
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        
        self.logger.info(f"Данные разделены: {len(train_df)} строк для обучения, {len(test_df)} строк для тестирования")
        
        # Создаем временные файлы для тренировочных и тестовых данных
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_file = os.path.join(self.output_dir, f'train_data_{timestamp}.csv')
        test_file = os.path.join(self.output_dir, f'test_data_{timestamp}.csv')
        
        train_df.to_csv(train_file)
        test_df.to_csv(test_file)
        
        # Параметры для моделей
        if self.use_grid_search_params and self.best_params_file and os.path.exists(self.best_params_file):
            self.logger.info(f"Загрузка лучших параметров из {self.best_params_file}")
            with open(self.best_params_file, 'r') as f:
                best_params = json.load(f)
                signal_params = best_params['signal_params']
                portfolio_params = best_params['portfolio_params']
                if 'risk_free_rate' in best_params:
                    self.risk_free_rate = best_params['risk_free_rate']
        else:
            # Параметры по умолчанию, если не используем Grid Search
            self.logger.info("Использование параметров по умолчанию")
            signal_params = {
                'weight_tech': 0.5,
                'weight_sentiment': 0.3,
                'weight_fundamental': 0.2,
                'threshold_buy': 0.5,
                'threshold_sell': -0.5
            }
            
            portfolio_params = {
                'min_rf_allocation': 0.25,
                'max_rf_allocation': 0.35
            }
        
        # 1. Обучение на тренировочных данных
        self.logger.info("Запуск генерации сигналов на тренировочных данных")
        train_signals_file = os.path.join(self.output_dir, f'train_signals_{timestamp}.csv')
        
        # Запускаем генератор сигналов на тренировочных данных
        signal_gen = SignalGenerator(
            input_file=train_file,
            **signal_params
        )
        train_signals_df = signal_gen.run_pipeline(
            output_file=train_signals_file,
            output_dir=os.path.join(self.output_dir, 'train_results')
        )
        
        self.logger.info("Запуск оптимизации портфеля на тренировочных данных")
        portfolio_opt = PortfolioOptimizer(
            input_file=train_signals_file,
            risk_free_rate=self.risk_free_rate,
            **portfolio_params
        )
        
        train_portfolio = portfolio_opt.run_pipeline(
            output_dir=os.path.join(self.output_dir, 'train_portfolio')
        )
        
        if train_portfolio is None:
            self.logger.error("Ошибка оптимизации портфеля на тренировочных данных")
            return None
        
        # 2. Генерация сигналов на тестовых данных (без изменения модели)
        self.logger.info("Запуск генерации сигналов на тестовых данных")
        test_signals_file = os.path.join(self.output_dir, f'test_signals_{timestamp}.csv')
        
        signal_gen = SignalGenerator(
            input_file=test_file,
            **signal_params
        )
        test_signals_df = signal_gen.run_pipeline(
            output_file=test_signals_file,
            output_dir=os.path.join(self.output_dir, 'test_results')
        )
        
        # 3. Применение модели портфеля из тренировки к тестовым данным
        self.logger.info("Запуск бэктеста на тестовых данных")
        backtester_obj = Backtester(
            input_file=test_signals_file,
            portfolio_weights=train_portfolio['weights'],
            start_date=test_start,
            end_date=test_end
        )
        
        test_results = backtester_obj.run_pipeline(
            output_dir=os.path.join(self.output_dir, 'test_backtest'),
            risk_free_rate=self.risk_free_rate
        )
        
        if test_results is None:
            self.logger.error("Ошибка при бэктесте на тестовых данных")
            return None
        
        # 4. Для сравнения делаем бэктест на тренировочных данных
        self.logger.info("Запуск бэктеста на тренировочных данных для сравнения")
        train_backtester = Backtester(
            input_file=train_signals_file,
            portfolio_weights=train_portfolio['weights'],
            start_date=train_start,
            end_date=train_end
        )
        
        train_backtest_results = train_backtester.run_pipeline(
            output_dir=os.path.join(self.output_dir, 'train_backtest'),
            risk_free_rate=self.risk_free_rate
        )
        
        # 5. Визуализация сравнения результатов
        self.logger.info("Создание сравнительной визуализации")
        
        plt.figure(figsize=(12, 8))
        
        # График кумулятивной доходности на тренировочных данных
        train_cum_returns = train_backtest_results['cumulative_returns']
        plt.plot(train_cum_returns.index, train_cum_returns, label='Тренировочный период', color='blue')
        
        # График кумулятивной доходности на тестовых данных
        test_cum_returns = test_results['cumulative_returns']
        plt.plot(test_cum_returns.index, test_cum_returns, label='Тестовый период', color='red')
        
        plt.title('Сравнение тренировочного и тестового периодов')
        plt.xlabel('Дата')
        plt.ylabel('Кумулятивная доходность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'train_vs_test_comparison.png'))
        plt.close()
        
        # 6. Создание сводного отчета
        self.logger.info("Создание сводного отчета")
        
        report = {
            'train_period': self.train_period,
            'test_period': self.test_period,
            'train_metrics': train_backtest_results['metrics'],
            'test_metrics': test_results['metrics'],
            'used_params': {
                'signal_params': signal_params,
                'portfolio_params': portfolio_params,
                'risk_free_rate': self.risk_free_rate
            }
        }
        
        # Сохраняем отчет в JSON
        with open(os.path.join(self.output_dir, 'honest_backtest_report.json'), 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        # Создаем markdown отчет для более удобного чтения
        with open(os.path.join(self.output_dir, 'honest_backtest_report.md'), 'w') as f:
            f.write("# Отчет о честном бэктесте\n\n")
            
            f.write("## Периоды\n")
            f.write(f"* Тренировочный период: {self.train_period[0]} - {self.train_period[1]}\n")
            f.write(f"* Тестовый период: {self.test_period[0]} - {self.test_period[1]}\n\n")
            
            f.write("## Метрики тренировочного периода\n")
            f.write(f"* Годовая доходность: {train_backtest_results['metrics']['annual_return']*100:.2f}%\n")
            f.write(f"* Годовая волатильность: {train_backtest_results['metrics']['annual_volatility']*100:.2f}%\n")
            f.write(f"* Коэффициент Шарпа: {train_backtest_results['metrics']['sharpe_ratio']:.2f}\n")
            f.write(f"* Максимальная просадка: {train_backtest_results['metrics']['max_drawdown']*100:.2f}%\n")
            f.write(f"* Винрейт: {train_backtest_results['metrics']['win_rate']*100:.2f}%\n\n")
            
            f.write("## Метрики тестового периода\n")
            f.write(f"* Годовая доходность: {test_results['metrics']['annual_return']*100:.2f}%\n")
            f.write(f"* Годовая волатильность: {test_results['metrics']['annual_volatility']*100:.2f}%\n")
            f.write(f"* Коэффициент Шарпа: {test_results['metrics']['sharpe_ratio']:.2f}\n")
            f.write(f"* Максимальная просадка: {test_results['metrics']['max_drawdown']*100:.2f}%\n")
            f.write(f"* Винрейт: {test_results['metrics']['win_rate']*100:.2f}%\n\n")
            
            f.write("## Использованные параметры\n")
            f.write("### Сигнальный генератор\n")
            for k, v in signal_params.items():
                f.write(f"* {k}: {v}\n")
            
            f.write("\n### Оптимизатор портфеля\n")
            for k, v in portfolio_params.items():
                f.write(f"* {k}: {v}\n")
            
            f.write(f"\nБезрисковая ставка: {self.risk_free_rate*100:.2f}%\n")
        
        # Удаляем временные файлы
        for temp_file in [train_file, test_file, train_signals_file, test_signals_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        self.logger.info(f"Отчет сохранен в {self.output_dir}")
        
        return report


def run_honest_backtest(
    data_file="/Users/aeshef/Documents/GitHub/kursach/data/df.csv",
    best_params_file=None,
    train_period=('2024-01-01', '2024-12-31'),
    test_period=('2025-01-01', '2025-06-30'),
    output_dir="/Users/aeshef/Documents/GitHub/kursach/data/honest_backtest",
    risk_free_rate=0.075,
    use_grid_search_params=True
):
    """
    Запускает честный бэктест на будущий период используя класс HonestBacktester
    
    Parameters:
    -----------
    data_file : str
        Путь к файлу с данными
    best_params_file : str, optional
        Путь к файлу с лучшими параметрами из Grid Search
    train_period : tuple
        Период обучения (начало, конец)
    test_period : tuple
        Период тестирования (начало, конец)
    output_dir : str
        Директория для сохранения результатов
    risk_free_rate : float
        Безрисковая ставка
    use_grid_search_params : bool
        Использовать ли параметры из Grid Search
        
    Returns:
    --------
    dict с результатами бэктеста
    """
    backtester = HonestBacktester(
        data_file=data_file,
        best_params_file=best_params_file,
        train_period=train_period,
        test_period=test_period,
        output_dir=output_dir,
        risk_free_rate=risk_free_rate,
        use_grid_search_params=use_grid_search_params
    )
    
    return backtester.run()
