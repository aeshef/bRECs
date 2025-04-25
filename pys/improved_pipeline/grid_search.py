import pandas as pd
import numpy as np
import itertools
import logging
import os
import json
from datetime import datetime
import importlib
import sys
import gc
from tqdm import tqdm
import concurrent.futures

from pys.porfolio_optimization.signal_generator import SignalGenerator, run_pipeline_signal_generator
from pys.porfolio_optimization.portfolio_optimizer import PortfolioOptimizer
from pys.porfolio_optimization.backtester import Backtester

# sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection')
# from private_info import BASE_PATH

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class GridSearch:
    def __init__(self, data_file, output_dir, log_level=logging.ERROR, n_jobs=1):
        """
        Реализация Grid Search для оптимизации параметров пайплайна
        
        Parameters:
        -----------
        data_file : str
            Путь к файлу с данными
        output_dir : str
            Директория для сохранения результатов
        log_level : logging.Level
            Уровень логирования (для уменьшения вывода рекомендуется ERROR)
        n_jobs : int
            Количество параллельных процессов (1 = без параллелизма)
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.log_level = log_level
        self.n_jobs = n_jobs
        self.results = []
        self.logger = logging.getLogger('GridSearch')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        
        self.logger.setLevel(log_level)
        
        # Проверяем, есть ли уже обработчики, чтобы избежать дублирования
        if not self.logger.handlers:
            file_handler = logging.FileHandler(os.path.join(output_dir, 'grid_search.log'))
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Добавляем вывод в консоль
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(file_formatter)
            self.logger.addHandler(console_handler)
            
        self.logger.info("GridSearch initialized with output directory: " + output_dir)

    def _test_combination(self, args):
        """Отдельная функция для тестирования одной комбинации параметров"""
        period_idx, start_date, end_date, signal_params_dict, portfolio_params_dict, rf_rate, test_id = args
        
        signals_file = os.path.join(self.output_dir, f'signals_temp_{test_id}.csv')
        
        try:
            # Генерация сигналов
            signal_gen = SignalGenerator(
                input_file=self.data_file,
                log_level=self.log_level,
                **signal_params_dict
            )
            
            signals_df = signal_gen.run_pipeline(
                output_file=signals_file,
                output_dir=None  # Отключаем запись промежуточных результатов
            )
            
            if signals_df is None:
                return None
                
            # Оптимизация портфеля
            portfolio_opt = PortfolioOptimizer(
                input_file=signals_file,
                risk_free_rate=rf_rate,
                log_level=self.log_level,
                **portfolio_params_dict
            )
            
            portfolio = portfolio_opt.run_pipeline(output_dir=None)
            
            if portfolio is None:
                return None
            
            # Бэктестирование
            backtester_obj = Backtester(
                input_file=signals_file,
                portfolio_weights=portfolio['weights'],
                start_date=start_date,
                end_date=end_date,
                log_level=self.log_level
            )
            
            results = backtester_obj.run_pipeline(
                output_dir=None,
                risk_free_rate=rf_rate
            )
            
            if results is None or 'metrics' not in results:
                return None
            
            metrics = results['metrics']
            result_entry = {
                'period_idx': period_idx,
                'start_date': start_date,
                'end_date': end_date,
                'risk_free_rate': rf_rate,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'annual_return': metrics['annual_return'],
                'annual_volatility': metrics['annual_volatility'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate']
            }
            
            for k, v in signal_params_dict.items():
                result_entry[f'signal_{k}'] = v
            for k, v in portfolio_params_dict.items():
                result_entry[f'portfolio_{k}'] = v
                
            return result_entry
            
        except Exception as e:
            self.logger.error(f"Ошибка при тестировании комбинации параметров (ID: {test_id}): {e}")
            return None
        finally:
            # Удаляем временный файл
            if os.path.exists(signals_file):
                try:
                    os.remove(signals_file)
                except:
                    pass
            
            # Явно очищаем память
            gc.collect()

    def run_grid_search(self, signal_params, portfolio_params, risk_free_rates=None, periods=None, chunk_size=10):
        """
        Запускает Grid Search по заданным параметрам с поддержкой чанкинга и контрольных точек
        
        Parameters:
        -----------
        signal_params : dict
            Словарь с параметрами для SignalGenerator в формате {param_name: [values]}
        portfolio_params : dict
            Словарь с параметрами для PortfolioOptimizer в формате {param_name: [values]}
        risk_free_rates : list, optional
            Список безрисковых ставок для тестирования
        periods : list, optional
            Список периодов для тестирования в формате [(start_date, end_date)]
        chunk_size : int
            Размер чанка для обработки и сохранения промежуточных результатов
            
        Returns:
        --------
        DataFrame с результатами и лучшими параметрами
        """
        if risk_free_rates is None:
            risk_free_rates = [0.075]
            
        if periods is None:
            periods = [('2024-01-01', '2024-12-31')]
        
        # Создаем список всех комбинаций параметров для тестирования
        combinations = []
        
        signal_keys = list(signal_params.keys())
        signal_values = list(signal_params.values())
        signal_combinations = list(itertools.product(*signal_values))
        
        portfolio_keys = list(portfolio_params.keys())
        portfolio_values = list(portfolio_params.values())
        portfolio_combinations = list(itertools.product(*portfolio_values))
        
        # Создаем все комбинации параметров для тестирования
        for period_idx, (start_date, end_date) in enumerate(periods):
            for signal_comb in signal_combinations:
                signal_params_dict = dict(zip(signal_keys, signal_comb))
                for portfolio_comb in portfolio_combinations:
                    portfolio_params_dict = dict(zip(portfolio_keys, portfolio_comb))
                    for rf_rate in risk_free_rates:
                        test_id = f"{period_idx}_{hash(str(signal_params_dict))}_{hash(str(portfolio_params_dict))}_{rf_rate}"
                        combinations.append((period_idx, start_date, end_date, signal_params_dict, portfolio_params_dict, rf_rate, test_id))
        
        total_tests = len(combinations)
        self.logger.info(f"Запуск Grid Search: {total_tests} комбинаций параметров")
        
        # Проверяем наличие контрольных точек
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'last_results.csv')
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Найдена контрольная точка. Загрузка предыдущих результатов...")
            checkpoint_results = pd.read_csv(checkpoint_path)
            self.results = checkpoint_results.to_dict('records')
            self.logger.info(f"Загружено {len(self.results)} результатов из контрольной точки")
        
        # Разбиваем комбинации на чанки
        chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]
        
        # Запускаем тестирование по чанкам
        for chunk_idx, chunk in enumerate(chunks):
            self.logger.info(f"Обработка чанка {chunk_idx+1}/{len(chunks)} ({len(chunk)} комбинаций)")
            
            # Используем ThreadPoolExecutor для параллельного выполнения
            if self.n_jobs > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    # Запускаем тесты в параллель с отображением прогресса
                    results = list(tqdm(executor.map(self._test_combination, chunk), total=len(chunk), desc=f"Чанк {chunk_idx+1}"))
                    
                    # Отфильтровываем None результаты и добавляем в общий список
                    valid_results = [r for r in results if r is not None]
                    self.results.extend(valid_results)
            else:
                # Последовательное выполнение
                for args in tqdm(chunk, desc=f"Чанк {chunk_idx+1}"):
                    result = self._test_combination(args)
                    if result is not None:
                        self.results.append(result)
            
            # Сохраняем промежуточные результаты
            if self.results:
                results_df = pd.DataFrame(self.results)
                checkpoint_file = os.path.join(self.output_dir, 'checkpoints', f'results_chunk_{chunk_idx}.csv')
                results_df.to_csv(checkpoint_file, index=False)
                
                # Обновляем последнюю контрольную точку
                results_df.to_csv(os.path.join(self.output_dir, 'checkpoints', 'last_results.csv'), index=False)
                
                self.logger.info(f"Сохранены промежуточные результаты: {len(self.results)} тестов")
            
            # Очищаем память
            gc.collect()
        
        # Финальная обработка результатов
        if not self.results:
            self.logger.error("Нет результатов Grid Search")
            return None, None
            
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        results_file = os.path.join(self.output_dir, 'grid_search_results.csv')
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Результаты сохранены в {results_file}")
        
        best_params = results_df.iloc[0].to_dict()
        best_params_file = os.path.join(self.output_dir, 'best_params.json')
        
        best_signal_params = {k.replace('signal_', ''): v for k, v in best_params.items() if k.startswith('signal_')}
        best_portfolio_params = {k.replace('portfolio_', ''): v for k, v in best_params.items() if k.startswith('portfolio_')}
        
        with open(best_params_file, 'w') as f:
            json.dump({
                'signal_params': best_signal_params,
                'portfolio_params': best_portfolio_params,
                'risk_free_rate': best_params['risk_free_rate'],
                'metrics': {
                    'sharpe_ratio': best_params['sharpe_ratio'],
                    'annual_return': best_params['annual_return'],
                    'annual_volatility': best_params['annual_volatility'],
                    'max_drawdown': best_params['max_drawdown'],
                    'win_rate': best_params['win_rate']
                }
            }, f, indent=4)
        
        self.logger.info(f"Лучшие параметры сохранены в {best_params_file}")
        
        return results_df, best_params

def run_grid_search_pipeline(
    data_file=f"{BASE_PATH}/data/df.csv",
    output_dir=f"{BASE_PATH}/data/grid_search_results",
    training_period=('2024-01-01', '2024-12-31'),
    n_jobs=4  # По умолчанию используем 4 потока
):
    """
    Запускает Grid Search для поиска оптимальных параметров пайплайна
    
    Parameters:
    -----------
    data_file : str
        Путь к файлу с данными
    output_dir : str
        Директория для сохранения результатов
    training_period : tuple
        Период обучения в формате (start_date, end_date)
    n_jobs : int
        Количество параллельных процессов
        
    Returns:
    --------
    (DataFrame, dict) - DataFrame с результатами и словарь с лучшими параметрами
    """
    # Упрощаем сетку параметров для ускорения поиска
    signal_params = {
        'weight_tech': [0.4, 0.5, 0.6],
        'weight_sentiment': [0.2, 0.3],
        'weight_fundamental': [0.1, 0.2],
        'threshold_buy': [0.2, 0.4, 0.6],  # Добавляем более низкий порог
        'threshold_sell': [-0.2, -0.4, -0.6]  # Добавляем более высокий порог
    }
    
    portfolio_params = {
        'min_rf_allocation': [0.25, 0.3],
        'max_rf_allocation': [0.35, 0.4]
    }
    
    risk_free_rates = [0.075]  # Упрощаем для скорости
    
    grid_search = GridSearch(
        data_file=data_file,
        output_dir=output_dir,
        log_level=logging.INFO,  # Используем INFO для лучшего мониторинга
        n_jobs=n_jobs
    )
    
    # Используем чанкинг для сохранения промежуточных результатов
    results_df, best_params = grid_search.run_grid_search(
        signal_params=signal_params,
        portfolio_params=portfolio_params,
        risk_free_rates=risk_free_rates,
        periods=[training_period],
        chunk_size=20  # Сохраняем результаты каждые 20 комбинаций
    )
    
    return results_df, best_params
