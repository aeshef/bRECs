import os
import datetime
import shutil
import json
import pandas as pd
import numpy as np
import logging
from pys.utils.logger import BaseLogger
import matplotlib.pyplot as plt

class PipelineExecutor(BaseLogger):
    """
    Выполняет и организует финальные этапы пайплайна инвестиционной стратегии.
    
    Класс выполняет:
    - Генерацию сигналов
    - Оптимизацию портфелей разных типов (лонг, шорт, комбинированный)
    - Выбор лучшего портфеля на основе метрик
    - Создание отчета о запуске
    
    Все результаты сохраняются в отдельной директории для каждого запуска.
    """
    
    def __init__(self, base_path, bond_results=None, name=None, strategy_profile='moderate',
                 min_position_weight=0.01, min_rf_allocation=0.3, max_rf_allocation=0.5, 
                 risk_free_rate=0.1, max_weight=0.15):
        """
        Инициализирует выполнитель пайплайна.
        
        Parameters:
        -----------
        base_path : str
            Базовый путь проекта
        bond_results : dict, optional
            Результаты выполнения run_bond_selection_with_kbd
        name : str, optional
            Дополнительное имя для директории запуска
        strategy_profile : str, optional
            Профиль стратегии ('aggressive', 'moderate', 'conservative')
        min_position_weight : float, optional
            Минимальный вес позиции в портфеле (позиции с меньшим весом удаляются)
        min_rf_allocation : float, optional
            Минимальная доля безрисковых активов
        max_rf_allocation : float, optional
            Максимальная доля безрисковых активов
        risk_free_rate : float, optional
            Безрисковая ставка доходности
        max_weight : float, optional
            Максимальный вес одной позиции в портфеле
        """
        super().__init__('PipelineExecutor')
        
        # Создаем идентификатор запуска с временной меткой
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name_suffix = f"_{name}" if name else ""
        self.run_id = f"run_{self.timestamp}{name_suffix}"
        
        self.base_path = base_path
        self.bond_results = bond_results
        self.bond_dataset_path = None  # Будет хранить путь к датасету бондов
        
        # Параметры стратегии
        self.strategy_profile = strategy_profile
        self.min_position_weight = min_position_weight
        self.min_rf_allocation = min_rf_allocation
        self.max_rf_allocation = max_rf_allocation
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        
        # Создаем директорию для текущего запуска
        self.run_dir = os.path.join(base_path, "data", "pipeline_runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Создаем поддиректории для разных этапов
        self.signals_dir = os.path.join(self.run_dir, "signals")
        self.portfolio_dir = os.path.join(self.run_dir, "portfolio")
        self.backtest_dir = os.path.join(self.run_dir, "backtest")
        self.shorts_dir = os.path.join(self.run_dir, "shorts_portfolio")
        self.combined_dir = os.path.join(self.run_dir, "combined_portfolio")
        self.final_dir = os.path.join(self.run_dir, "final_portfolio")
        
        for directory in [self.signals_dir, self.portfolio_dir, self.backtest_dir, 
                          self.shorts_dir, self.combined_dir, self.final_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Создана структура директорий для запуска {self.run_id}")
        
        # Инициализируем параметры и результаты
        self.signal_params = {}
        self.portfolio_params = {}
        self.short_params = {}
        self.combined_params = {}
        self.results = {
            'bond_portfolio': None,
            'signals': None,
            'standard_portfolio': None,
            'short_portfolio': None,
            'combined_portfolio': None,
            'best_portfolio': None
        }
    
    def process_bond_pipeline(self, start_date, end_date, min_bonds=20, max_threshold=99, 
                         strategy_profile=None, kbd_yield_adjustment=-3.0, 
                         update_kbd_data=True, excluded_issuers=None, n_bonds=5, kbd_data=None,
                         # Добавляем недостающие параметры, которые нужны для run_bond_selection_with_kbd
                         weighting_strategy=None, portfolio_stability=0.7,
                         use_kbd_recommendations=True, override_params=None,
                         kbd_duration_flexibility=1.5, max_adjustment_iterations=3,
                         output_format='all'):
        """
        Запускает полный цикл обработки облигаций - от парсинга до создания портфеля.
        
        Parameters:
        -----------
        start_date, end_date : str
            Диапазон дат для анализа
        min_bonds : int
            Минимальное количество облигаций в выборке
        max_threshold : int
            Максимальный порог для фильтрации облигаций
        strategy_profile : str, optional
            Профиль стратегии ('aggressive', 'moderate', 'conservative')
        kbd_yield_adjustment : float
            Корректировка доходности КБД
        update_kbd_data : bool
            Флаг обновления данных КБД
        excluded_issuers : list, optional
            Список исключаемых эмитентов
        n_bonds : int
            Количество облигаций для включения в портфель
        kbd_data : pandas.DataFrame, optional
            Предварительно загруженные данные КБД
        
        Returns:
        --------
        dict
            Результаты обработки облигаций
        """
        from pys.data_collection.bonds_parser import run_bond_pipeline  # Исправленный импорт
        from pys.data_collection.bonds_kbd_pipeline import run_bond_selection_with_kbd
        
        self.logger.info("Запуск полного цикла обработки облигаций")
        
        # Создаем поддиректории для облигаций внутри текущего запуска пайплайна
        bonds_dir = os.path.join(self.run_dir, "bonds")
        bonds_analysis_dir = os.path.join(bonds_dir, "analysis")
        bonds_portfolio_dir = os.path.join(bonds_dir, "portfolio")
        bonds_reports_dir = os.path.join(bonds_dir, "reports")
        bonds_viz_dir = os.path.join(bonds_dir, "viz")
        
        for directory in [bonds_dir, bonds_analysis_dir, bonds_portfolio_dir, 
                        bonds_reports_dir, bonds_viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Создана структура директорий для облигаций в {bonds_dir}")
        
        # Использовать стратегию из параметров объекта, если не указана
        if strategy_profile is None:
            strategy_profile = self.strategy_profile
            
        # Исключаемые эмитенты по умолчанию
        if excluded_issuers is None:
            excluded_issuers = ['ВТБ', 'Мечел']
        
        try:
            # 1. Запуск run_bond_pipeline для получения отфильтрованного набора данных
            pipeline_results = run_bond_pipeline(
                base_path=self.base_path,
                start_date=start_date,
                end_date=end_date,
                min_bonds=min_bonds,
                max_threshold=max_threshold
            )
            
            # Копируем датасет в директорию пайплайна для полноты
            if 'dataset_path' in pipeline_results and os.path.exists(pipeline_results['dataset_path']):
                dataset_copy_path = os.path.join(bonds_dir, "bonds_dataset.csv")
                shutil.copy2(pipeline_results['dataset_path'], dataset_copy_path)
                self.logger.info(f"Датасет облигаций скопирован в {dataset_copy_path}")
                self.bond_dataset_path = dataset_copy_path
            else:
                self.bond_dataset_path = pipeline_results.get("dataset_path")
            
            self.logger.info(f"Обработка облигаций: оптимальный порог {pipeline_results['optimal_threshold']}%")
            
            # 2. Запуск run_bond_selection_with_kbd для выбора облигаций в портфель
            # Передаем новые пути для сохранения результатов внутри пайплайна
            bond_results = run_bond_selection_with_kbd(
                base_path=self.base_path,
                dataset_path=self.bond_dataset_path,
                n_bonds=n_bonds,
                min_bonds=min_bonds,
                weighting_strategy=weighting_strategy,
                portfolio_stability=portfolio_stability,
                use_kbd_recommendations=use_kbd_recommendations,
                override_params=override_params,
                start_date=start_date,
                end_date=end_date,
                update_kbd_data=update_kbd_data,
                strategy_profile=strategy_profile,
                kbd_yield_adjustment=kbd_yield_adjustment,
                kbd_duration_flexibility=kbd_duration_flexibility,
                max_adjustment_iterations=max_adjustment_iterations,
                excluded_issuers=excluded_issuers,
                output_format=output_format,
                kbd_data=kbd_data,
                output_dirs={
                    'portfolios': bonds_portfolio_dir,
                    'analysis': bonds_analysis_dir,
                    'reports': bonds_reports_dir,
                    'viz': bonds_viz_dir
                }
            )
            
            # Перенаправляем пути к файлам в структуру пайплайна
            if bond_results and 'paths' in bond_results:
                # Сохраняем копию портфеля в корневой директории пайплайна для удобного доступа
                if 'portfolio_path' in bond_results['paths'] and os.path.exists(bond_results['paths']['portfolio_path']):
                    bond_portfolio_copy = os.path.join(self.run_dir, "bond_portfolio.csv")
                    shutil.copy2(bond_results['paths']['portfolio_path'], bond_portfolio_copy)
                    bond_results['paths']['pipeline_portfolio_path'] = bond_portfolio_copy
                    self.logger.info(f"Портфель облигаций дополнительно скопирован в {bond_portfolio_copy}")
            
            # Сохраняем результаты для дальнейшего использования
            self.bond_results = bond_results
            
            # Сохраняем метаданные о выполненном процессе
            metadata = {
                'run_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'strategy_profile': strategy_profile,
                'parameters': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'min_bonds': min_bonds,
                    'max_threshold': max_threshold,
                    'kbd_yield_adjustment': kbd_yield_adjustment,
                    'n_bonds': n_bonds,
                    'excluded_issuers': excluded_issuers
                },
                'results': {
                    'optimal_threshold': pipeline_results.get('optimal_threshold'),
                    'bond_count': len(bond_results.get('portfolio', [])) if bond_results.get('portfolio') is not None else 0,
                    'weighted_yield': bond_results.get('stats', {}).get('weighted_yield'),
                    'weighted_duration': bond_results.get('stats', {}).get('weighted_duration')
                },
                'paths': {
                    'dataset_path': self.bond_dataset_path,
                    'portfolio_path': bond_results.get('paths', {}).get('portfolio_path'),
                    'report_path': bond_results.get('paths', {}).get('report_path')
                }
            }
            
            # Сохраняем метаданные в JSON
            metadata_path = os.path.join(bonds_dir, "bond_pipeline_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return bond_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке облигаций: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def copy_bond_portfolio(self, bond_results=None):
        """
        Копирует результаты выбора облигаций.
        
        Parameters:
        -----------
        bond_results : dict, optional
            Результаты выполнения run_bond_selection_with_kbd
        
        Returns:
        --------
        str
            Путь к скопированному портфелю облигаций
        """
        bond_results = bond_results or self.bond_results
        
        if not bond_results:
            self.logger.warning("Результаты выбора облигаций не предоставлены")
            return None
        
        if 'paths' not in bond_results or 'portfolio_path' not in bond_results['paths']:
            self.logger.warning("В результатах выбора облигаций отсутствуют необходимые пути")
            return None
        
        # Путь к оригинальному портфелю облигаций
        bond_portfolio_path = bond_results['paths']['portfolio_path']
        
        # Проверяем существование файла
        if not os.path.exists(bond_portfolio_path):
            self.logger.warning(f"Файл портфеля облигаций не найден: {bond_portfolio_path}")
            return None
        
        # Копируем портфель облигаций
        bond_portfolio_copy = os.path.join(self.run_dir, "bond_portfolio.csv")
        shutil.copy2(bond_portfolio_path, bond_portfolio_copy)
        
        self.logger.info(f"Скопирован портфель облигаций: {bond_portfolio_copy}")
        self.results['bond_portfolio'] = bond_portfolio_copy
        
        return bond_portfolio_copy
    
    def generate_signals(self, weight_tech=0.5, weight_sentiment=0.3, weight_fundamental=0.2,
                     threshold_buy=0.5, threshold_sell=-0.5, top_pct=0.3, 
                     save_ticker_visualizations=False):
        """
        Запускает генерацию сигналов.
        
        Parameters:
        -----------
        weight_tech : float
            Вес технических сигналов
        weight_sentiment : float
            Вес сентимент-сигналов
        weight_fundamental : float
            Вес фундаментальных сигналов
        
        Returns:
        --------
        dict
            Результаты генерации сигналов
        """
        from pys.porfolio_optimization.signal_generator import run_pipeline_signal_generator
        
        self.signal_params = {
            "weight_tech": weight_tech,
            "weight_sentiment": weight_sentiment,
            "weight_fundamental": weight_fundamental,
            "threshold_buy": threshold_buy,
            "threshold_sell": threshold_sell,
            "top_pct": top_pct
        }   
        
        self.logger.info(f"Запуск генерации сигналов с параметрами: {self.signal_params}")
        
        # Запускаем генерацию сигналов
        signals = run_pipeline_signal_generator(
        weight_tech=weight_tech,
        weight_sentiment=weight_sentiment,
        weight_fundamental=weight_fundamental,
        output_dir=self.signals_dir,  # Указываем директорию для визуализаций
        save_ticker_visualizations=save_ticker_visualizations
        )
        
        # Получаем путь к сгенерированным сигналам
        signals_file = os.path.join(self.base_path, "data", "signals.csv")
        
        # Копируем файл сигналов в нашу директорию
        signals_copy = os.path.join(self.signals_dir, "signals.csv")
        if os.path.exists(signals_file):
            shutil.copy2(signals_file, signals_copy)
            self.logger.info(f"Скопированы сигналы: {signals_copy}")
        else:
            self.logger.warning(f"Файл сигналов не найден: {signals_file}")
        
        # Также создаем файл с параметрами сигналов
        params_file = os.path.join(self.signals_dir, "signal_params.json")
        with open(params_file, 'w') as f:
            json.dump(self.signal_params, f, indent=4)
        
        self.results['signals'] = {
            'signals_path': signals_copy if os.path.exists(signals_file) else None,
            'params_path': params_file
        }
        
        return signals
    
    def optimize_standard_portfolio(self, tickers_list, risk_free_rate=None, min_rf_allocation=None, 
                                    max_rf_allocation=None, max_weight=None, include_short_selling=False,
                                    optimization='markowitz'):
        """
        Запускает оптимизацию стандартного портфеля (только длинные позиции).
        
        Parameters:
        -----------
        tickers_list : list
            Список тикеров
        risk_free_rate : float, optional
            Безрисковая ставка
        min_rf_allocation, max_rf_allocation : float, optional
            Минимальная и максимальная доля безрисковых активов
        max_weight : float, optional
            Максимальный вес одного актива
        
        Returns:
        --------
        dict
            Результаты оптимизации портфеля
        """
        from pys.porfolio_optimization.portfolio_optimizer import run_all_optimization_models
        
        # Использовать параметры из объекта, если не указаны
        risk_free_rate = risk_free_rate or self.risk_free_rate
        min_rf_allocation = min_rf_allocation or self.min_rf_allocation
        max_rf_allocation = max_rf_allocation or self.max_rf_allocation
        max_weight = max_weight or self.max_weight
        
        self.portfolio_params = {
            "risk_free_rate": risk_free_rate,
            "min_rf_allocation": min_rf_allocation,
            "max_rf_allocation": max_rf_allocation,
            "max_weight": max_weight,
            "include_short_selling": include_short_selling,
            "optimization": optimization  # Add this parameter
        }
        
        self.logger.info(f"Запуск оптимизации стандартного портфеля с параметрами: {self.portfolio_params}")
        
        # Проверяем наличие портфеля облигаций
        if not self.results['bond_portfolio']:
            self.logger.warning("Портфель облигаций не найден, копирование...")
            self.copy_bond_portfolio()
        
        bond_portfolio_path = self.results['bond_portfolio']
        
        try:
            # Запускаем оптимизацию
            portfolio_results = run_all_optimization_models(
                base_path=self.base_path,
                tickers_list=tickers_list,
                risk_free_rate=risk_free_rate,
                min_rf_allocation=min_rf_allocation,
                max_rf_allocation=max_rf_allocation,
                max_weight=max_weight,
                risk_free_portfolio_file=bond_portfolio_path,
                include_short_selling=include_short_selling,
                optimization=optimization  # Add this parameter
            )
            
            # Копируем результаты после выполнения
            for root, dirs, files in os.walk(os.path.join(self.base_path, 'data', 'portfolio')):
                for file in files:
                    if file.endswith('.png') or file.endswith('.csv') or file.endswith('.json'):
                        src_path = os.path.join(root, file)
                        # Создаем относительный путь
                        rel_path = os.path.relpath(root, os.path.join(self.base_path, 'data', 'portfolio'))
                        dest_dir = os.path.join(self.portfolio_dir, rel_path)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, file)
                        shutil.copy2(src_path, dest_path)
            
            # Фильтруем портфель по минимальному весу позиций
            for model in portfolio_results:
                if 'weights' in portfolio_results[model]:
                    portfolio_results[model]['weights'] = self._filter_small_weights(
                        portfolio_results[model]['weights'], 
                        self.min_position_weight
                    )
            
            # Сохраняем параметры оптимизации
            params_file = os.path.join(self.portfolio_dir, "portfolio_params.json")
            with open(params_file, 'w') as f:
                json.dump(self.portfolio_params, f, indent=4)
            
            self.results['standard_portfolio'] = portfolio_results
            self.logger.info(f"Стандартный портфель успешно оптимизирован и сохранен в {self.portfolio_dir}")
            
            return portfolio_results
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации стандартного портфеля: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_short_portfolio(self, risk_free_rate=None, train_period=('2024-01-01', '2024-12-31'), 
                           test_period=('2025-01-01', '2025-06-30'), best_params_file=None, verify_with_honest_backtest=True):
        """
        Создает портфель с короткими позициями.
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            Безрисковая ставка
        train_period, test_period : tuple
            Периоды для обучения и тестирования
        best_params_file : str, optional
            Путь к файлу с лучшими параметрами
        
        Returns:
        --------
        dict
            Результаты создания портфеля с короткими позициями
        """
        # Использовать параметры из объекта, если не указаны
        risk_free_rate = risk_free_rate or self.risk_free_rate
        
        self.short_params = {
            "risk_free_rate": risk_free_rate,
            "train_period": train_period,
            "test_period": test_period,
            "include_short_selling": True,
            "verify_with_honest_backtest" : verify_with_honest_backtest
        }
        
        self.logger.info(f"Запуск создания портфеля с короткими позициями с параметрами: {self.short_params}")
        
        # Проверяем существование директории для честного бэктеста
        honest_dir = os.path.join(self.shorts_dir, 'honest_backtest')
        os.makedirs(honest_dir, exist_ok=True)
        
        try:
            # Если путь к best_params_file не указан, используем значение по умолчанию
            if not best_params_file:
                best_params_file = os.path.join(self.base_path, "data", "grid_search", "best_params.json")
            
            # Запускаем построение портфеля с использованием встроенного метода
            portfolio_results = self.build_production_portfolio(
                data_file=f"{self.base_path}/data/df.csv",
                output_dir=self.shorts_dir,
                risk_free_rate=risk_free_rate,
                best_params_file=best_params_file,
                include_short_selling=True,
                verify_with_honest_backtest=verify_with_honest_backtest,
                train_period=train_period,
                test_period=test_period
            )
            
            # Фильтруем портфель по минимальному весу позиций
            if 'production_portfolio' in portfolio_results and 'weights' in portfolio_results['production_portfolio']:
                portfolio_results['production_portfolio']['weights'] = self._filter_small_weights(
                    portfolio_results['production_portfolio']['weights'], 
                    self.min_position_weight
                )
            
            # Сохраняем параметры
            params_file = os.path.join(self.shorts_dir, "short_portfolio_params.json")
            with open(params_file, 'w') as f:
                json.dump(self.short_params, f, indent=4)
            
            self.results['short_portfolio'] = portfolio_results
            self.logger.info(f"Портфель с короткими позициями успешно создан и сохранен в {self.shorts_dir}")
            
            return portfolio_results
        except Exception as e:
            self.logger.error(f"Ошибка при создании портфеля с короткими позициями: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_combined_portfolio(self, tickers_list, risk_free_rate=None, min_rf_allocation=None, 
                        max_rf_allocation=None, max_weight=None, long_ratio=0.7,
                        include_short_selling=True):
        """
        Создает комбинированный портфель с длинными и короткими позициями.
        """
        from pys.porfolio_optimization.portfolio_optimizer import run_all_optimization_models, PortfolioOptimizer
        
        # Использовать параметры из объекта, если не указаны
        risk_free_rate = risk_free_rate or self.risk_free_rate
        min_rf_allocation = min_rf_allocation or self.min_rf_allocation
        max_rf_allocation = max_rf_allocation or self.max_rf_allocation
        max_weight = max_weight or self.max_weight
        
        self.combined_params = {
            "risk_free_rate": risk_free_rate,
            "min_rf_allocation": min_rf_allocation,
            "max_rf_allocation": max_rf_allocation,
            "max_weight": max_weight,
            "long_ratio": long_ratio,
            "include_short_selling": include_short_selling  # Use the parameter instead of hardcoded True
        }
        
        self.logger.info(f"Запуск создания комбинированного портфеля с параметрами: {self.combined_params}")
        
        # Проверяем наличие портфеля облигаций
        if not self.results['bond_portfolio']:
            self.logger.warning("Портфель облигаций не найден, копирование...")
            self.copy_bond_portfolio()
        
        bond_portfolio_path = self.results['bond_portfolio']
        
        try:
            input_file = os.path.join(self.base_path, "data", "signals.csv")
            if not os.path.exists(input_file):
                input_file = os.path.join(self.signals_dir, "signals.csv")
                if not os.path.exists(input_file):
                    self.logger.error("Не найден файл с сигналами для оптимизации портфеля")
                    return None
            
            # Create optimizer with short positions support
            optimizer = PortfolioOptimizer(
                input_file=input_file,
                risk_free_rate=risk_free_rate,
                min_rf_allocation=min_rf_allocation,
                max_rf_allocation=max_rf_allocation,
                max_weight=max_weight,
                optimization='markowitz', ## поч
                risk_free_portfolio_file=bond_portfolio_path,
                include_short_selling=include_short_selling  # Use the parameter instead of hardcoded True
            )

            
            # Загружаем данные
            optimizer.load_data()
            
            # Подготовка доходностей с учетом коротких позиций
            returns = optimizer.prepare_returns_with_shorts()
            
            if returns is None or returns.empty:
                self.logger.error("Не удалось подготовить доходности для комбинированного портфеля")
                return None
            
            # Создаем границы для весов: от -max_weight до +max_weight
            n_assets = len(returns.columns)
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
            
            # Оптимизация портфеля
            optimization_result = optimizer.optimize_portfolio(
                returns=returns,
                risk_free_rate=risk_free_rate,
                constrained=True,
                bounds=bounds
            )
            
            # Проверяем успешность оптимизации
            if optimization_result is None or not optimization_result.get('success', False):
                self.logger.error(f"Не удалось оптимизировать комбинированный портфель")
                return None
                
            # Вот ключевой момент! Нужно вызвать calculate_final_portfolio()
            # чтобы инициализировать portfolio_performance
            rf_allocation = (min_rf_allocation + max_rf_allocation) / 2
            optimizer.calculate_final_portfolio(rf_allocation=rf_allocation)
            
            # Проверяем, что portfolio_performance не None
            if optimizer.portfolio_performance is None:
                self.logger.error("Не удалось рассчитать характеристики портфеля")
                return None
                
            # Получаем оптимальные веса и rf_allocation
            optimal_weights = optimizer.optimal_weights
            rf_allocation = optimizer.portfolio_performance.get('rf_allocation', rf_allocation)
            
            # Создаем словарь весов с тикерами
            tickers = returns.columns
            weights_dict = dict(zip(tickers, optimal_weights))
            
            # Разделяем на длинные и короткие позиции
            long_positions = {ticker: weight for ticker, weight in weights_dict.items() if weight > 0}
            short_positions = {ticker: weight for ticker, weight in weights_dict.items() if weight < 0}
            
            # Сумма всех положительных весов и всех отрицательных весов по модулю
            sum_long = sum(long_positions.values())
            sum_short = sum(abs(weight) for weight in short_positions.values())
            
            # Проверяем, чтобы не было деления на ноль
            if sum_long <= 0 or sum_short <= 0:
                self.logger.warning(f"Невозможно создать сбалансированный лонг-шорт портфель: long={sum_long}, short={sum_short}")
                # Создаем безрисковый портфель как запасной вариант
                return {
                    'weights': {'RISK_FREE': 1.0},
                    'expected_return': risk_free_rate,
                    'expected_volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'risk_free_rate': risk_free_rate,
                    'rf_allocation': 1.0
                }
            
            # Нормализуем длинные и короткие позиции
            long_positions = {ticker: weight / sum_long for ticker, weight in long_positions.items()}
            short_positions = {ticker: weight / sum_short for ticker, weight in short_positions.items()}
            
            # Применяем коэффициент распределения между длинными и короткими позициями
            long_weight = long_ratio * (1 - rf_allocation)
            short_weight = (1 - long_ratio) * (1 - rf_allocation)
            
            # Итоговые веса длинных и коротких позиций с учетом пропорции
            final_long_positions = {ticker: weight * long_weight for ticker, weight in long_positions.items()}
            final_short_positions = {ticker: -weight * short_weight for ticker, weight in short_positions.items()}
            
            # Объединяем все позиции в один словарь
            combined_weights = {**final_long_positions, **final_short_positions}
            
            # Добавляем безрисковую часть
            combined_weights['RISK_FREE'] = rf_allocation
            
            # Фильтруем по минимальному весу
            combined_weights = self._filter_small_weights(combined_weights, self.min_position_weight)
            
            # Создаем новый словарь portfolio_performance для комбинированного портфеля
            combined_portfolio = {
                'weights': combined_weights,
                'long_positions': final_long_positions,
                'short_positions': final_short_positions,
                'risk_free_rate': risk_free_rate,
                'rf_allocation': rf_allocation,
                'optimization_model': 'combined'
            }
            
            # Расчет метрик комбинированного портфеля
            portfolio_return = 0
            portfolio_vol = 0
            
            # Для длинных позиций 
            if final_long_positions:
                long_weights = np.array(list(final_long_positions.values()))
                long_tickers = list(final_long_positions.keys())
                
                if len(long_tickers) > 0 and all(ticker in returns.columns for ticker in long_tickers):
                    long_returns = returns[long_tickers]
                    
                    long_return = np.sum(long_returns.mean() * long_weights) * 252
                    long_vol = np.sqrt(np.dot(long_weights.T, np.dot(long_returns.cov() * 252, long_weights)))
                    
                    portfolio_return += long_return
                    portfolio_vol += long_vol**2
            
            # Для коротких позиций (инвертируем доходность)
            if final_short_positions:
                short_weights = np.array(list(final_short_positions.values()))
                short_tickers = list(final_short_positions.keys())
                
                if len(short_tickers) > 0 and all(ticker in returns.columns for ticker in short_tickers):
                    short_returns = returns[short_tickers]
                    
                    # Для коротких позиций инвертируем доходность
                    short_return = -np.sum(short_returns.mean() * short_weights) * 252
                    short_vol = np.sqrt(np.dot(short_weights.T, np.dot(short_returns.cov() * 252, short_weights)))
                    
                    portfolio_return += short_return
                    portfolio_vol += short_vol**2
            
            # Учитываем безрисковую часть
            portfolio_return += rf_allocation * risk_free_rate
            
            # Итоговая волатильность - корень из суммы квадратов
            portfolio_vol = np.sqrt(portfolio_vol) if portfolio_vol > 0 else 0.0001
            
            # Рассчитываем коэффициент Шарпа
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Обновляем метрики в результатах
            combined_portfolio['expected_return'] = portfolio_return
            combined_portfolio['expected_volatility'] = portfolio_vol
            combined_portfolio['sharpe_ratio'] = sharpe_ratio
            
            # Визуализация комбинированного портфеля
            self._visualize_combined_portfolio(combined_portfolio, self.combined_dir)
            
            # Сохраняем параметры
            params_file = os.path.join(self.combined_dir, "combined_portfolio_params.json")
            with open(params_file, 'w') as f:
                json.dump(self.combined_params, f, indent=4)
            
            # Сохраняем веса в CSV
            weights_df = pd.DataFrame([
                {'Ticker': ticker, 'Weight': weight, 'Type': 'LONG' if weight > 0 else 'SHORT' if ticker != 'RISK_FREE' else 'RISK_FREE'}
                for ticker, weight in combined_weights.items()
            ])
            weights_df.to_csv(os.path.join(self.combined_dir, 'combined_weights.csv'), index=False)
            
            self.results['combined_portfolio'] = combined_portfolio
            self.logger.info(f"Комбинированный портфель успешно создан и сохранен в {self.combined_dir}")
            
            return combined_portfolio
        except Exception as e:
            self.logger.error(f"Ошибка при создании комбинированного портфеля: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    
    def _filter_small_weights(self, weights_dict, min_weight=0.01):
        """
        Фильтрует позиции с весом меньше указанного минимального значения.
        
        Parameters:
        -----------
        weights_dict : dict
            Словарь с весами портфеля
        min_weight : float
            Минимальный вес позиции
        
        Returns:
        --------
        dict
            Отфильтрованный словарь весов
        """
        # Отделяем RISK_FREE от остальных позиций
        rf_weight = weights_dict.get('RISK_FREE', 0)
        other_weights = {ticker: weight for ticker, weight in weights_dict.items() if ticker != 'RISK_FREE'}
        
        # Отфильтровываем слишком малые веса
        significant_weights = {ticker: weight for ticker, weight in other_weights.items() if abs(weight) >= min_weight}
        
        # Рассчитываем сумму отфильтрованных весов
        removed_weight = sum(weight for ticker, weight in other_weights.items() if abs(weight) < min_weight)
        
        # Если есть отфильтрованные веса, перераспределяем их пропорционально остальным
        if removed_weight != 0 and significant_weights:
            # Распределяем удаленный вес пропорционально существующим весам
            sum_significant = sum(abs(weight) for weight in significant_weights.values())
            
            for ticker in significant_weights:
                significant_weights[ticker] += (abs(significant_weights[ticker]) / sum_significant) * removed_weight
        
        # Добавляем RISK_FREE обратно
        if rf_weight > 0:
            significant_weights['RISK_FREE'] = rf_weight
        
        return significant_weights
    
    def _visualize_combined_portfolio(self, portfolio, output_dir):
        """
        Создает визуализацию для комбинированного портфеля.
        
        Parameters:
        -----------
        portfolio : dict
            Информация о портфеле
        output_dir : str
            Директория для сохранения визуализаций
        """
        if 'weights' not in portfolio:
            self.logger.error("Нет данных о весах для визуализации комбинированного портфеля")
            return
        
        weights = portfolio['weights']
        
        # Подготавливаем данные для визуализации
        long_positions = {ticker: weight for ticker, weight in weights.items() 
                         if weight > 0 and ticker != 'RISK_FREE'}
        short_positions = {ticker: abs(weight) for ticker, weight in weights.items() 
                          if weight < 0}
        
        # 1. Сначала создаем общую диаграмму
        plt.figure(figsize=(12, 8))
        
        # Объединяем длинные и короткие позиции для визуализации
        all_positions = {}
        for ticker, weight in long_positions.items():
            all_positions[f"{ticker} (LONG)"] = weight
        
        for ticker, weight in short_positions.items():
            all_positions[f"{ticker} (SHORT)"] = -weight  # Отрицательное значение для отображения короткой позиции
        
        # Добавляем безрисковую часть
        if 'RISK_FREE' in weights:
            all_positions['RISK_FREE'] = weights['RISK_FREE']
        
        # Сортируем по значению для лучшей визуализации
        sorted_positions = {k: v for k, v in sorted(all_positions.items(), key=lambda item: abs(item[1]), reverse=True)}
        
        # Цвета для разных типов позиций
        colors = []
        labels = []
        values = []
        
        for label, value in sorted_positions.items():
            labels.append(label)
            values.append(abs(value))  # Используем абсолютное значение для размера в пироге
            
            if 'RISK_FREE' in label:
                colors.append('gray')
            elif 'LONG' in label:
                colors.append('green')
            else:  # SHORT
                colors.append('red')
        
        # Создаем пай-чарт
        plt.pie(
            values, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
        )
        plt.axis('equal')
        plt.title('Комбинированный портфель (длинные и короткие позиции)')
        
        plt.savefig(os.path.join(output_dir, 'combined_portfolio_pie.png'))
        plt.close()
        
        # 2. Создаем столбчатую диаграмму для лучшей визуализации распределения весов
        plt.figure(figsize=(14, 10))
        
        # Подготавливаем данные для столбчатой диаграммы
        bar_labels = []
        bar_values = []
        bar_colors = []
        
        # Собираем безрисковую часть
        if 'RISK_FREE' in weights:
            bar_labels.append('RISK_FREE')
            bar_values.append(weights['RISK_FREE'])
            bar_colors.append('gray')
        
        # Собираем длинные позиции (сортируем по убыванию)
        for ticker, weight in sorted(long_positions.items(), key=lambda x: x[1], reverse=True):
            bar_labels.append(f"{ticker} (LONG)")
            bar_values.append(weight)
            bar_colors.append('green')
        
        # Собираем короткие позиции (сортируем по убыванию)
        for ticker, weight in sorted(short_positions.items(), key=lambda x: x[1], reverse=True):
            bar_labels.append(f"{ticker} (SHORT)")
            bar_values.append(-weight)  # Отрицательное значение для визуализации
            bar_colors.append('red')
        
        # Создаем столбчатую диаграмму
        bars = plt.barh(bar_labels, bar_values, color=bar_colors)
        
        # Добавляем подписи значений
        for bar in bars:
            width = bar.get_width()
            label_position = width if width > 0 else width - 0.02
            align = 'left' if width > 0 else 'right'
            plt.text(
                label_position, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.2%}',
                va='center',
                ha=align,
                fontweight='bold'
            )
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.title('Распределение весов в комбинированном портфеле', fontsize=16)
        plt.xlabel('Вес в портфеле', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_portfolio_bars.png'))
        plt.close()
        
        # 3. Сохраняем текстовый отчет с метриками
        with open(os.path.join(output_dir, 'combined_portfolio_metrics.txt'), 'w') as f:
            f.write("МЕТРИКИ КОМБИНИРОВАННОГО ПОРТФЕЛЯ\n")
            f.write("==================================\n\n")
            f.write(f"Ожидаемая годовая доходность: {portfolio['expected_return']*100:.2f}%\n")
            f.write(f"Ожидаемая годовая волатильность: {portfolio['expected_volatility']*100:.2f}%\n")
            f.write(f"Коэффициент Шарпа: {portfolio['sharpe_ratio']:.2f}\n")
            f.write(f"Безрисковая ставка: {portfolio['risk_free_rate']*100:.2f}%\n")
            f.write(f"Доля безрисковых активов: {portfolio['rf_allocation']*100:.2f}%\n\n")
            
            f.write("Распределение длинных и коротких позиций:\n")
            f.write(f"- Общая доля длинных позиций: {sum(long_positions.values())*100:.2f}%\n")
            f.write(f"- Общая доля коротких позиций: {sum(short_positions.values())*100:.2f}%\n")
            f.write(f"- Безрисковая часть: {weights.get('RISK_FREE', 0)*100:.2f}%\n\n")
            
            f.write("Длинные позиции:\n")
            for ticker, weight in sorted(long_positions.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {ticker}: {weight*100:.2f}%\n")
            
            f.write("\nКороткие позиции:\n")
            for ticker, weight in sorted(short_positions.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {ticker}: {-weight*100:.2f}%\n")
    
    def select_best_portfolio(self, metrics_priority=None, min_sharpe=0, prefer_standard=False, force_portfolio_type=None):
        """
        Выбирает лучший портфель на основе метрик и стратегии риска.
        Копирует выбранный портфель в директорию final_portfolio.
        
        Parameters:
        -----------
        metrics_priority : list, optional
            Приоритет метрик для выбора ['sharpe', 'return', 'volatility']
        min_sharpe : float, optional
            Минимальный допустимый коэффициент Шарпа
        prefer_standard : bool, optional
            Предпочитать ли стандартный портфель при прочих равных
        force_portfolio_type : str, optional
            Принудительно выбрать тип портфеля ('standard', 'short', 'combined')
        
        Returns:
        --------
        dict
            Информация о выбранном лучшем портфеле
        """
        # Проверяем, созданы ли необходимые портфели
        available_portfolios = {}
        
        # Стандартный портфель
        if self.results['standard_portfolio'] and 'markowitz' in self.results['standard_portfolio']:
            standard_portfolio = self.results['standard_portfolio']['markowitz']
            if standard_portfolio:
                available_portfolios['standard'] = {
                    'portfolio': standard_portfolio,
                    'source_dir': self.portfolio_dir,
                    'metrics': {
                        'sharpe_ratio': standard_portfolio.get('sharpe_ratio', 0),
                        'expected_return': standard_portfolio.get('expected_return', 0),
                        'expected_volatility': standard_portfolio.get('expected_volatility', 1)
                    }
                }
        
        # Портфель с короткими позициями
        if self.results['short_portfolio'] and 'production_portfolio' in self.results['short_portfolio']:
            short_portfolio = self.results['short_portfolio']['production_portfolio']
            if short_portfolio:
                short_metrics = {}
                if 'production_backtest' in self.results['short_portfolio']:
                    backtest_metrics = self.results['short_portfolio']['production_backtest'].get('metrics', {})
                    short_metrics = {
                        'sharpe_ratio': backtest_metrics.get('sharpe_ratio', 0),
                        'expected_return': backtest_metrics.get('annual_return', 0),
                        'expected_volatility': backtest_metrics.get('annual_volatility', 1)
                    }
                else:
                    short_metrics = {
                        'sharpe_ratio': short_portfolio.get('sharpe_ratio', 0),
                        'expected_return': short_portfolio.get('expected_return', 0),
                        'expected_volatility': short_portfolio.get('expected_volatility', 1)
                    }
                
                available_portfolios['short'] = {
                    'portfolio': short_portfolio,
                    'source_dir': self.shorts_dir,
                    'metrics': short_metrics
                }
        
        # Комбинированный портфель
        if self.results['combined_portfolio']:
            combined_portfolio = self.results['combined_portfolio']
            if combined_portfolio:
                available_portfolios['combined'] = {
                    'portfolio': combined_portfolio,
                    'source_dir': self.combined_dir,
                    'metrics': {
                        'sharpe_ratio': combined_portfolio.get('sharpe_ratio', 0),
                        'expected_return': combined_portfolio.get('expected_return', 0),
                        'expected_volatility': combined_portfolio.get('expected_volatility', 1)
                    }
                }
        
        if not available_portfolios:
            self.logger.error("Не найдено ни одного портфеля для выбора")
            return None
        
        # Выбор портфеля - новая логика с использованием параметров
        best_portfolio = None
        best_portfolio_name = None
        
        # 1. Принудительный выбор типа портфеля, если указан
        if force_portfolio_type and force_portfolio_type in available_portfolios:
            best_portfolio = available_portfolios[force_portfolio_type]
            best_portfolio_name = force_portfolio_type
            self.logger.info(f"Принудительно выбран портфель типа '{force_portfolio_type}'")
        
        # 2. Выбор на основе приоритета метрик, если указан
        elif metrics_priority:
            best_score = -float('inf')
            for name, portfolio_info in available_portfolios.items():
                # Пропускаем портфели с Шарпом ниже минимального
                if min_sharpe > 0 and portfolio_info['metrics']['sharpe_ratio'] < min_sharpe:
                    continue
                    
                # Пропускаем не-стандартные портфели при prefer_standard=True
                if prefer_standard and 'standard' in available_portfolios and name != 'standard':
                    continue
                    
                # Рассчитываем оценку на основе приоритета метрик
                score = 0
                for i, metric in enumerate(metrics_priority):
                    weight = len(metrics_priority) - i  # Более приоритетные метрики получают больший вес
                    if metric == 'sharpe':
                        score += weight * portfolio_info['metrics']['sharpe_ratio']
                    elif metric == 'return':
                        score += weight * portfolio_info['metrics']['expected_return']
                    elif metric == 'volatility':
                        # Меньшая волатильность лучше, поэтому инвертируем
                        score += weight * (1 / (portfolio_info['metrics']['expected_volatility'] or 0.0001))
                        
                if score > best_score:
                    best_score = score
                    best_portfolio = portfolio_info
                    best_portfolio_name = name
                    
            if best_portfolio:
                self.logger.info(f"Выбран портфель '{best_portfolio_name}' на основе приоритета метрик {metrics_priority}")
        
        # 3. Если выбор еще не сделан, используем логику на основе профиля стратегии
        if best_portfolio is None:
            if self.strategy_profile == 'aggressive':
                # Предоставляем все варианты, но выбираем портфель с максимальной доходностью
                max_return = -float('inf')
                for name, portfolio_info in available_portfolios.items():
                    if portfolio_info['metrics']['expected_return'] > max_return:
                        max_return = portfolio_info['metrics']['expected_return']
                        best_portfolio = portfolio_info
                        best_portfolio_name = name
                
                self.logger.info(f"Агрессивная стратегия: выбран портфель '{best_portfolio_name}' с максимальной доходностью {max_return*100:.2f}%")
            
            elif self.strategy_profile == 'moderate':
                # Выбираем между комбинированным и длинным с лучшим Шарпом
                moderate_portfolios = {name: info for name, info in available_portfolios.items() 
                                    if name in ['standard', 'combined']}
                
                if not moderate_portfolios and 'short' in available_portfolios:
                    moderate_portfolios = {'short': available_portfolios['short']}
                
                max_sharpe = -float('inf')
                for name, portfolio_info in moderate_portfolios.items():
                    if portfolio_info['metrics']['sharpe_ratio'] > max_sharpe:
                        max_sharpe = portfolio_info['metrics']['sharpe_ratio']
                        best_portfolio = portfolio_info
                        best_portfolio_name = name
                
                self.logger.info(f"Умеренная стратегия: выбран портфель '{best_portfolio_name}' с лучшим Шарпом {max_sharpe:.2f}")
            
            else:  # conservative
                # Только длинный портфель с большей долей безрисковой части
                if 'standard' in available_portfolios:
                    best_portfolio = available_portfolios['standard']
                    best_portfolio_name = 'standard'
                    self.logger.info(f"Консервативная стратегия: выбран стандартный портфель с Шарпом {best_portfolio['metrics']['sharpe_ratio']:.2f}")
                else:
                    # Если нет стандартного, берем любой доступный с наименьшей волатильностью
                    min_vol = float('inf')
                    for name, portfolio_info in available_portfolios.items():
                        if portfolio_info['metrics']['expected_volatility'] < min_vol:
                            min_vol = portfolio_info['metrics']['expected_volatility']
                            best_portfolio = portfolio_info
                            best_portfolio_name = name
                    
                    self.logger.info(f"Консервативная стратегия: выбран портфель '{best_portfolio_name}' с минимальной волатильностью {min_vol*100:.2f}%")
        
        if best_portfolio is None:
            self.logger.error("Не удалось выбрать лучший портфель")
            return None
        
        # Копируем файлы выбранного портфеля в final_portfolio
        source_dir = best_portfolio['source_dir']
        
        try:
            # Копируем все файлы из директории выбранного портфеля
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    src_path = os.path.join(root, file)
                    # Создаем относительный путь
                    rel_path = os.path.relpath(root, source_dir)
                    dest_dir = os.path.join(self.final_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, file)
                    shutil.copy2(src_path, dest_path)
            
            # Создаем README с объяснением выбора
            with open(os.path.join(self.final_dir, 'README.md'), 'w') as f:
                f.write(f"# Лучший портфель: {best_portfolio_name.upper()}\n\n")
                f.write(f"Профиль стратегии: **{self.strategy_profile}**\n\n")
                f.write("## Ключевые метрики портфеля\n\n")
                f.write(f"- Ожидаемая доходность: {best_portfolio['metrics']['expected_return']*100:.2f}%\n")
                f.write(f"- Ожидаемая волатильность: {best_portfolio['metrics']['expected_volatility']*100:.2f}%\n")
                f.write(f"- Коэффициент Шарпа: {best_portfolio['metrics']['sharpe_ratio']:.2f}\n\n")
                
                f.write("## Объяснение выбора\n\n")
                if force_portfolio_type:
                    f.write(f"Портфель выбран принудительно (force_portfolio_type='{force_portfolio_type}').\n\n")
                elif metrics_priority:
                    f.write(f"Портфель выбран на основе приоритета метрик: {', '.join(metrics_priority)}.\n")
                    if min_sharpe > 0:
                        f.write(f"Применен фильтр минимального коэффициента Шарпа: {min_sharpe}.\n")
                    if prefer_standard:
                        f.write("Применено предпочтение стандартного портфеля.\n")
                    f.write("\n")
                elif self.strategy_profile == 'aggressive':
                    f.write("Для агрессивного профиля выбран портфель с максимальной ожидаемой доходностью. ")
                    f.write("Этот портфель может иметь более высокий риск, но предлагает потенциально более высокую доходность.\n\n")
                elif self.strategy_profile == 'moderate':
                    f.write("Для умеренного профиля выбран портфель с оптимальным соотношением риска и доходности (наивысший коэффициент Шарпа). ")
                    f.write("Рассматривались только стандартный и комбинированный портфели для большей стабильности.\n\n")
                else:  # conservative
                    f.write("Для консервативного профиля выбран портфель с наименьшим риском. ")
                    f.write("Предпочтение отдавалось стандартному портфелю с большей долей безрисковых активов.\n\n")
                
                f.write("## Доступные альтернативы\n\n")
                
                f.write("| Портфель | Ожидаемая доходность | Волатильность | Шарп |\n")
                f.write("|----------|----------------------|---------------|------|\n")
                
                for name, info in available_portfolios.items():
                    metrics = info['metrics']
                    f.write(f"| {name.capitalize()} | {metrics['expected_return']*100:.2f}% | {metrics['expected_volatility']*100:.2f}% | {metrics['sharpe_ratio']:.2f} |\n")
            
            # Сохраняем информацию о лучшем портфеле
            self.results['best_portfolio'] = {
                'type': best_portfolio_name,
                'portfolio': best_portfolio['portfolio'],
                'metrics': best_portfolio['metrics']
            }
            
            self.logger.info(f"Лучший портфель ({best_portfolio_name}) скопирован в {self.final_dir}")
            
            return self.results['best_portfolio']
            
        except Exception as e:
            self.logger.error(f"Ошибка при копировании лучшего портфеля: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    
    def create_summary_report(self, include_charts=True, include_metrics=True, include_weights=True, report_format='md'):
        """
        Создает итоговый отчет о запуске.
        
        Returns:
        --------
        str
            Путь к созданному отчету
        """
        summary_path = os.path.join(self.run_dir, "pipeline_summary.md")
        
        self.logger.info(f"Создание итогового отчета о запуске")
        
        try:
            with open(summary_path, 'w') as f:
                f.write(f"# Отчет о запуске пайплайна\n\n")
                f.write(f"**Дата и время запуска:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Идентификатор запуска:** {self.run_id}\n")
                f.write(f"**Профиль стратегии:** {self.strategy_profile}\n\n")
                
                # Параметры
                f.write("## Параметры\n\n")
                
                if self.signal_params:
                    f.write("### Параметры генерации сигналов\n")
                    for key, value in self.signal_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if self.portfolio_params:
                    f.write("### Параметры оптимизации стандартного портфеля\n")
                    for key, value in self.portfolio_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if self.short_params:
                    f.write("### Параметры портфеля с короткими позициями\n")
                    for key, value in self.short_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if self.combined_params:
                    f.write("### Параметры комбинированного портфеля\n")
                    for key, value in self.combined_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if include_metrics:
                    f.write("## Результаты\n\n")
                    
                    # Стандартный портфель
                    standard_portfolio = self.results.get('standard_portfolio')
                    if standard_portfolio and 'markowitz' in standard_portfolio:
                        markowitz_data = standard_portfolio['markowitz']
                        f.write("### Стандартный портфель (Markowitz)\n\n")
                        if 'expected_return' in markowitz_data:
                            f.write(f"- Ожидаемая доходность: {markowitz_data['expected_return']*100:.2f}%\n")
                        if 'expected_volatility' in markowitz_data:
                            f.write(f"- Ожидаемая волатильность: {markowitz_data['expected_volatility']*100:.2f}%\n")
                        if 'sharpe_ratio' in markowitz_data:
                            f.write(f"- Коэффициент Шарпа: {markowitz_data['sharpe_ratio']:.2f}\n\n")
                    
                    # Портфель с короткими позициями
                    short_portfolio = self.results.get('short_portfolio')
                    if short_portfolio and 'production_backtest' in short_portfolio:
                        short_metrics = short_portfolio['production_backtest']['metrics']
                        f.write("### Портфель с короткими позициями\n\n")
                        if 'annual_return' in short_metrics:
                            f.write(f"- Годовая доходность: {short_metrics['annual_return']*100:.2f}%\n")
                        if 'sharpe_ratio' in short_metrics:
                            f.write(f"- Коэффициент Шарпа: {short_metrics['sharpe_ratio']:.2f}\n")
                        if 'max_drawdown' in short_metrics:
                            f.write(f"- Максимальная просадка: {short_metrics['max_drawdown']*100:.2f}%\n\n")
                    
                    # Комбинированный портфель
                    combined_portfolio = self.results.get('combined_portfolio')
                    if combined_portfolio:
                        f.write("### Комбинированный портфель\n\n")
                        if 'expected_return' in combined_portfolio:
                            f.write(f"- Ожидаемая доходность: {combined_portfolio['expected_return']*100:.2f}%\n")
                        if 'expected_volatility' in combined_portfolio:
                            f.write(f"- Ожидаемая волатильность: {combined_portfolio['expected_volatility']*100:.2f}%\n")
                        if 'sharpe_ratio' in combined_portfolio:
                            f.write(f"- Коэффициент Шарпа: {combined_portfolio['sharpe_ratio']:.2f}\n\n")
                    
                    # Лучший портфель
                    best_portfolio = self.results.get('best_portfolio')
                    if best_portfolio:
                        f.write("### ЛУЧШИЙ ПОРТФЕЛЬ\n\n")
                        f.write(f"**Тип портфеля: {best_portfolio['type'].upper()}**\n\n")
                        metrics = best_portfolio['metrics']
                        f.write(f"- Ожидаемая доходность: {metrics['expected_return']*100:.2f}%\n")
                        f.write(f"- Ожидаемая волатильность: {metrics['expected_volatility']*100:.2f}%\n")
                        f.write(f"- Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}\n\n")

                # Weights section - conditionally included
                if include_weights and self.results.get('best_portfolio'):
                    f.write("## Веса в итоговом портфеле\n\n")
                    best_portfolio = self.results['best_portfolio']['portfolio']
                    if 'weights' in best_portfolio:
                        f.write("| Актив | Вес |\n")
                        f.write("|-------|-----|\n")
                        
                        for ticker, weight in sorted(best_portfolio['weights'].items(), key=lambda x: abs(x[1]), reverse=True):
                            if ticker != 'rf_details':
                                direction = "" if weight >= 0 or ticker == 'RISK_FREE' else " (SHORT)"
                                f.write(f"| {ticker}{direction} | {weight*100:.2f}% |\n")
                        f.write("\n")
            
                # Charts section - conditionally included
                if include_charts:
                    f.write("## Графики\n\n")
                    # Reference to charts in the final portfolio directory
                    f.write("Визуализации доступны в директории финального портфеля:\n")
                    f.write(f"`{self.final_dir}`\n\n")
                    
                    # Расположение результатов
                    f.write("## Расположение результатов\n\n")
                    f.write(f"Все результаты этого запуска сохранены в директории:\n")
                    f.write(f"`{self.run_dir}`\n\n")
                    
                    f.write("### Структура директорий\n\n")
                    f.write("```\n")
                    f.write(f"{self.run_id}/\n")
                    f.write("├── signals/            # Сигналы для акций\n")
                    f.write("├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)\n")
                    f.write("├── shorts_portfolio/   # Портфель с короткими позициями\n")
                    f.write("├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)\n")
                    f.write("├── backtest/           # Результаты бэктестов\n")
                    f.write("├── final_portfolio/    # Лучший выбранный портфель\n")
                    f.write("└── bond_portfolio.csv  # Портфель облигаций\n")
                    f.write("```\n")

                if report_format == 'html':
                    # This would require additional implementation to convert MD to HTML
                    # For now, just add a note
                    html_path = os.path.join(self.run_dir, "pipeline_summary.html")
                    self.logger.info(f"HTML format requested but not implemented yet. Using Markdown format.")
            
            self.logger.info(f"Итоговый отчет создан: {summary_path}")
            return summary_path
        except Exception as e:
            self.logger.error(f"Ошибка при создании итогового отчета: {e}")
            return None

    def build_production_portfolio(self, data_file, output_dir, risk_free_rate=0.075, 
                                best_params_file=None, include_short_selling=False,
                                verify_with_honest_backtest=True, 
                                train_period=('2024-01-01', '2024-12-31'),
                                test_period=('2025-01-01', '2025-06-30')):
        """
        Комплексная функция построения и проверки инвестиционного портфеля
        
        Parameters:
        -----------
        data_file : str
            Путь к файлу с данными
        output_dir : str
            Директория для сохранения результатов
        risk_free_rate : float
            Безрисковая ставка
        best_params_file : str, optional
            Путь к файлу с оптимальными параметрами (из Grid Search)
        include_short_selling : bool
            Включать ли короткие позиции в портфель (по умолчанию False)
        verify_with_honest_backtest : bool
            Проверять ли стратегию с помощью HonestBacktester
        train_period, test_period : tuple
            Периоды для проверки с помощью HonestBacktester
        
        Returns:
        --------
        dict с результатами всех этапов
        """
        import os
        import pandas as pd
        import json
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        try:
            # 1. Проверка с помощью HonestBacktester (если нужно)
            if verify_with_honest_backtest:
                self.logger.info("Запуск проверки с HonestBacktester")
                
                honest_output_dir = os.path.join(output_dir, 'honest_backtest')
                os.makedirs(honest_output_dir, exist_ok=True)
                
                from pys.improved_pipeline.honest_backtest import HonestBacktester
                
                backtester = HonestBacktester(
                    data_file=data_file,
                    best_params_file=best_params_file,
                    train_period=train_period,
                    test_period=test_period,
                    output_dir=honest_output_dir,
                    risk_free_rate=risk_free_rate
                )
                
                honest_results = backtester.run()
                results['honest_backtest'] = honest_results
                
                if honest_results and 'test_metrics' in honest_results:
                    self.logger.info(f"Результаты честного бэктеста:")
                    self.logger.info(f"Доходность в тесте: {honest_results['test_metrics']['annual_return']*100:.2f}%")
                    self.logger.info(f"Шарп в тесте: {honest_results['test_metrics']['sharpe_ratio']:.2f}")
                    
                    # Если результаты плохие, можно добавить предупреждение
                    if honest_results['test_metrics']['sharpe_ratio'] < 0.5:
                        self.logger.warning("Низкий коэффициент Шарпа в тестовом периоде! Стратегия может быть неэффективна.")
            
            # 2. Построение финального портфеля
            self.logger.info(f"Построение финального портфеля на всех данных {'с поддержкой коротких позиций' if include_short_selling else 'только длинные позиции'}")
            
            production_dir = os.path.join(output_dir, 'production_portfolio')
            os.makedirs(production_dir, exist_ok=True)
            
            # Загружаем параметры из файла или используем дефолтные
            if best_params_file and os.path.exists(best_params_file):
                with open(best_params_file, 'r') as f:
                    best_params = json.load(f)
                    signal_params = best_params.get('signal_params', {})
                    portfolio_params = best_params.get('portfolio_params', {})
            else:
                # Параметры по умолчанию
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
            
            # a) Генерация сигналов
            from pys.porfolio_optimization.signal_generator import SignalGenerator
            
            signals_dir = os.path.join(production_dir, 'signals')
            os.makedirs(signals_dir, exist_ok=True)
            
            signal_gen = SignalGenerator(
                input_file=data_file,
                **signal_params
            )
            
            signals_file = os.path.join(signals_dir, 'production_signals.csv')
            signals_df = signal_gen.run_pipeline(
                output_file=signals_file,
                output_dir=signals_dir
            )
            
            # б) Оптимизация портфеля
            from pys.porfolio_optimization.portfolio_optimizer import PortfolioOptimizer
            
            portfolio_dir = os.path.join(production_dir, 'portfolio')
            os.makedirs(portfolio_dir, exist_ok=True)
            
            # Используем стандартный оптимизатор или модифицированный для коротких позиций
            if include_short_selling:
                self.logger.info("Используется оптимизатор с поддержкой коротких позиций")
                portfolio_opt = PortfolioOptimizer(
                    input_file=signals_file,
                    risk_free_rate=risk_free_rate,
                    include_short_selling=True,  # Включаем поддержку коротких позиций
                    **portfolio_params
                )
            else:
                self.logger.info("Используется стандартный оптимизатор (только длинные позиции)")
                portfolio_opt = PortfolioOptimizer(
                    input_file=signals_file,
                    risk_free_rate=risk_free_rate,
                    **portfolio_params
                )
            
            final_portfolio = portfolio_opt.run_pipeline(
                output_dir=portfolio_dir
            )
            
            # Если не удалось построить портфель, создаем безрисковый портфель
            if final_portfolio is None:
                self.logger.warning("Не удалось создать портфель с сигналами. Создаем безрисковый портфель.")
                final_portfolio = {
                    'weights': {'RISK_FREE': 1.0},
                    'expected_return': risk_free_rate,
                    'expected_volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'risk_free_rate': risk_free_rate,
                    'rf_allocation': 1.0,
                    'optimization_model': 'markowitz',
                    'rf_details': {}
                }
            
            results['production_portfolio'] = final_portfolio
            
            # в) Бэктест на всем периоде для оценки производительности
            if final_portfolio and 'weights' in final_portfolio:
                backtest_dir = os.path.join(production_dir, 'backtest')
                os.makedirs(backtest_dir, exist_ok=True)
                
                from pys.porfolio_optimization.backtester import Backtester
                
                backtester_obj = Backtester(
                    input_file=signals_file,
                    portfolio_weights=final_portfolio['weights']
                )
                
                backtest_results = backtester_obj.run_pipeline(
                    output_dir=backtest_dir,
                    risk_free_rate=risk_free_rate
                )
                
                results['production_backtest'] = backtest_results
            else:
                self.logger.error("Не удалось получить веса портфеля для бэктестирования")
                # Создаем пустые результаты бэктеста
                results['production_backtest'] = {
                    'metrics': {
                        'sharpe_ratio': 0.0,
                        'annual_return': risk_free_rate,
                        'max_drawdown': 0.0,
                        'annual_volatility': 0.0,
                        'win_rate': 0.5
                    },
                    'returns': pd.DataFrame(),
                    'cumulative_returns': pd.Series()
                }
            
            # 3. Создание итогового отчета
            with open(os.path.join(output_dir, 'production_summary.md'), 'w') as f:
                f.write("# Итоговый отчет о построении инвестиционного портфеля\n\n")
                f.write(f"**Режим портфеля:** {'С поддержкой коротких позиций' if include_short_selling else 'Только длинные позиции'}\n\n")
                
                f.write("## 1. Проверка стратегии\n")
                if verify_with_honest_backtest and 'honest_backtest' in results:
                    hb_results = results['honest_backtest']
                    f.write("### Результаты честного бэктеста\n")
                    f.write(f"* Тренировочный период: {train_period[0]} - {train_period[1]}\n")
                    f.write(f"* Тестовый период: {test_period[0]} - {test_period[1]}\n\n")
                    
                    if 'train_metrics' in hb_results:
                        tm = hb_results['train_metrics']
                        f.write("#### Тренировочный период\n")
                        f.write(f"* Годовая доходность: {tm['annual_return']*100:.2f}%\n")
                        f.write(f"* Шарп: {tm['sharpe_ratio']:.2f}\n")
                        f.write(f"* Макс. просадка: {tm['max_drawdown']*100:.2f}%\n\n")
                    
                    if 'test_metrics' in hb_results:
                        tm = hb_results['test_metrics']
                        f.write("#### Тестовый период\n")
                        f.write(f"* Годовая доходность: {tm['annual_return']*100:.2f}%\n")
                        f.write(f"* Шарп: {tm['sharpe_ratio']:.2f}\n")
                        f.write(f"* Макс. просадка: {tm['max_drawdown']*100:.2f}%\n\n")
                else:
                    f.write("Проверка с честным бэктестом не проводилась\n\n")
                
                f.write("## 2. Финальный портфель\n")
                if 'production_portfolio' in results:
                    pp = results['production_portfolio']
                    f.write(f"* Ожидаемая доходность: {pp['expected_return']*100:.2f}%\n")
                    f.write(f"* Ожидаемая волатильность: {pp['expected_volatility']*100:.2f}%\n")
                    f.write(f"* Коэффициент Шарпа: {pp['sharpe_ratio']:.2f}\n")
                    f.write(f"* Безрисковая ставка: {pp['risk_free_rate']*100:.2f}%\n")
                    f.write(f"* Доля безрисковых активов: {pp.get('rf_allocation', 0)*100:.2f}%\n\n")
                    
                    f.write("### Состав портфеля\n")
                    f.write("| Актив | Вес |\n")
                    f.write("|-------|-----|\n")
                    
                    weights = {k: v for k, v in pp['weights'].items() if k != 'rf_details'}
                    for ticker, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
                        if abs(weight) >= 0.01:  # Показываем только значимые позиции
                            # Показываем направление позиции для коротких позиций
                            direction = "SHORT" if weight < 0 else "LONG"
                            f.write(f"| {ticker} ({direction}) | {weight*100:.2f}% |\n")
                
                f.write("\n## 3. Рекомендации\n")
                f.write("* Портфель рекомендуется пересматривать ежеквартально\n")
                f.write("* Следует контролировать корреляцию активов\n")
                f.write("* При изменении рыночных условий возможно потребуется перенастройка параметров\n")
                if include_short_selling:
                    f.write("* Короткие позиции требуют более тщательного управления рисками\n")
                    f.write("* Регулярно проверяйте стоимость заимствования для коротких позиций\n")
            
            self.logger.info(f"Итоговый отчет сохранен в {output_dir}/production_summary.md")
            
            return results
        except Exception as e:
            self.logger.error(f"Ошибка при построении портфеля: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Создаем fallback-результаты в случае ошибки
            fallback_results = {
                'error': str(e),
                'production_backtest': {
                    'metrics': {
                        'sharpe_ratio': 0.0,
                        'annual_return': risk_free_rate,
                        'max_drawdown': 0.0,
                        'annual_volatility': 0.0
                    }
                },
                'production_portfolio': {
                    'weights': {'RISK_FREE': 1.0},
                    'expected_return': risk_free_rate,
                    'expected_volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'risk_free_rate': risk_free_rate,
                    'rf_allocation': 1.0
                }
            }
            return fallback_results

    

    def run_pipeline(self, tickers_list, bond_results=None, strategy_profile=None,
               signal_params=None, standard_portfolio_params=None, 
               short_portfolio_params=None, combined_portfolio_params=None,
               portfolio_controls=None, backtest_params=None,
               # Дополнительные параметры для детальной настройки
               select_portfolio_params=None, report_params=None,
               optimization_params=None, visualization_params=None):
        """
        Запускает все этапы пайплайна последовательно с полной настройкой параметров.
        
        Parameters:
        -----------
        tickers_list : list
            Список тикеров
        bond_results : dict, optional
            Результаты выполнения run_bond_selection_with_kbd
        strategy_profile : str, optional
            Профиль стратегии ('aggressive', 'moderate', 'conservative')
            
        # Параметры для разных этапов
        signal_params : dict, optional
            Параметры для генерации сигналов:
            - weight_tech (float): вес технических сигналов
            - weight_sentiment (float): вес сентимент-сигналов
            - weight_fundamental (float): вес фундаментальных сигналов
            - threshold_buy (float): порог для сигнала покупки
            - threshold_sell (float): порог для сигнала продажи
            - top_pct (float): процент лучших акций для shortlist
            - save_ticker_visualizations (bool): сохранять визуализации по тикерам
            
        standard_portfolio_params : dict, optional
            Параметры для стандартного портфеля:
            - risk_free_rate (float): безрисковая ставка
            - min_rf_allocation (float): минимальная доля безрисковых активов
            - max_rf_allocation (float): максимальная доля безрисковых активов
            - max_weight (float): максимальный вес одной позиции
            - include_short_selling (bool): включать ли короткие позиции
            
        short_portfolio_params : dict, optional
            Параметры для портфеля с короткими позициями:
            - risk_free_rate (float): безрисковая ставка
            - train_period (tuple): период обучения (начало, конец)
            - test_period (tuple): период тестирования (начало, конец)
            - best_params_file (str): путь к файлу с лучшими параметрами
            - verify_with_honest_backtest (bool): проверять с честным бэктестом
            
        combined_portfolio_params : dict, optional
            Параметры для комбинированного портфеля:
            - risk_free_rate (float): безрисковая ставка
            - min_rf_allocation (float): минимальная доля безрисковых активов
            - max_rf_allocation (float): максимальная доля безрисковых активов
            - max_weight (float): максимальный вес одной позиции
            - long_ratio (float): соотношение длинных/коротких позиций
            - include_short_selling (bool): включать ли короткие позиции
            
        portfolio_controls : dict, optional
            Контроль запуска различных портфелей:
            - run_standard_portfolio (bool): запускать стандартный портфель
            - run_short_portfolio (bool): запускать портфель с короткими позициями
            - run_combined_portfolio (bool): запускать комбинированный портфель
            - override_risk_profile (bool): игнорировать профиль риска
            
        backtest_params : dict, optional
            Параметры для бэктестирования:
            - train_period (tuple): период обучения (начало, конец)
            - test_period (tuple): период тестирования (начало, конец)
            - risk_free_rate (float): безрисковая ставка
            - use_grid_search_params (bool): использовать параметры из Grid Search
            
        select_portfolio_params : dict, optional
            Параметры для выбора лучшего портфеля:
            - metrics_priority (list): приоритет метрик ['sharpe', 'return', 'volatility']
            - min_sharpe (float): минимальный допустимый коэффициент Шарпа
            - prefer_standard (bool): предпочитать стандартный портфель
            - force_portfolio_type (str): принудительно выбрать тип портфеля
            
        report_params : dict, optional
            Параметры для создания отчета:
            - include_charts (bool): включать графики в отчет
            - include_metrics (bool): включать метрики в отчет
            - include_weights (bool): включать веса в отчет
            - report_format (str): формат отчета ('md', 'html')
            
        optimization_params : dict, optional
            Параметры оптимизации портфеля:
            - optimization (str): модель оптимизации ('markowitz', 'black_litterman')
            - tau (float): параметр неуверенности для Black-Litterman
            - views (dict): субъективные прогнозы 
            - view_confidences (dict): уверенность в прогнозах
            - market_caps (dict): рыночные капитализации
            
        visualization_params : dict, optional
            Параметры визуализации:
            - plot_style (str): стиль графиков ('seaborn', 'ggplot', etc.)
            - chart_size (tuple): размер графиков в дюймах
            - dpi (int): разрешение графиков
            - color_scheme (str): цветовая схема
            - save_formats (list): форматы сохранения графиков ['png', 'svg', 'pdf']
            
        Returns:
        --------
        dict
            Результаты выполнения всех этапов пайплайна
        """
        if strategy_profile:
            self.strategy_profile = strategy_profile
            
        self.logger.info(f"Запуск пайплайна (ID: {self.run_id}, профиль: {self.strategy_profile})")
        
        # Применяем визуализационные параметры, если переданы
        if visualization_params:
            if 'plot_style' in visualization_params:
                plt.style.use(visualization_params['plot_style'])
            # Другие параметры визуализации можно применить аналогично
        
        # Настройки запуска портфелей по умолчанию в зависимости от профиля риска
        default_controls = {
            'run_standard_portfolio': True,
            'run_short_portfolio': self.strategy_profile == 'aggressive',
            'run_combined_portfolio': self.strategy_profile in ['moderate', 'aggressive'],
            'override_risk_profile': False
        }
        
        # Обновляем настройки запуска, если они предоставлены
        controls = default_controls.copy()
        if portfolio_controls:
            controls.update(portfolio_controls)
        
        # 1. Копируем портфель облигаций
        self.copy_bond_portfolio(bond_results)
        
        # 2. Генерируем сигналы с параметрами
        signal_args = {}
        if signal_params:
            # Извлекаем все параметры для generate_signals
            for param in ['weight_tech', 'weight_sentiment', 'weight_fundamental', 
                        'threshold_buy', 'threshold_sell', 'top_pct']:
                if param in signal_params:
                    signal_args[param] = signal_params[param]
                    
        self.generate_signals(**signal_args)
        
        # 3. Создаем портфели в соответствии с настройками
        if controls['run_standard_portfolio']:
            self.logger.info(f"Профиль {self.strategy_profile}: создание стандартного портфеля")
            standard_args = {}
            if standard_portfolio_params:
                # Извлекаем все параметры для optimize_standard_portfolio
                for param in ['risk_free_rate', 'min_rf_allocation', 'max_rf_allocation', 
                            'max_weight', 'include_short_selling']:
                    if param in standard_portfolio_params:
                        standard_args[param] = standard_portfolio_params[param]
                        
            # Передаем параметры оптимизации, если они переданы
            if optimization_params and 'optimization' in optimization_params:
                standard_args['optimization'] = optimization_params['optimization']
                
            self.optimize_standard_portfolio(tickers_list, **standard_args)
        
        if controls['run_combined_portfolio']:
            self.logger.info(f"Профиль {self.strategy_profile}: создание комбинированного портфеля")
            combined_args = {}
            if combined_portfolio_params:
                # Извлекаем все параметры для create_combined_portfolio
                for param in ['risk_free_rate', 'min_rf_allocation', 'max_rf_allocation', 
                            'max_weight', 'long_ratio', 'include_short_selling']:
                    if param in combined_portfolio_params:
                        combined_args[param] = combined_portfolio_params[param]
                        
            self.create_combined_portfolio(tickers_list, **combined_args)
        
        if controls['run_short_portfolio']:
            self.logger.info(f"Профиль {self.strategy_profile}: создание портфеля с короткими позициями")
            short_args = {}
            if short_portfolio_params:
                # Извлекаем все параметры для create_short_portfolio
                for param in ['risk_free_rate', 'train_period', 'test_period', 'best_params_file', 
                            'verify_with_honest_backtest']:
                    if param in short_portfolio_params:
                        short_args[param] = short_portfolio_params[param]
                        
            self.create_short_portfolio(**short_args)
        
        # 4. Выбираем лучший портфель
        select_args = {}
        if select_portfolio_params:
            # Параметры для select_best_portfolio
            for param in ['metrics_priority', 'min_sharpe', 'prefer_standard', 'force_portfolio_type']:
                if param in select_portfolio_params:
                    select_args[param] = select_portfolio_params[param]
                    
        self.select_best_portfolio(**select_args)
        
        # 5. Создаем итоговый отчет
        report_args = {}
        if report_params:
            # Параметры для create_summary_report
            for param in ['include_charts', 'include_metrics', 'include_weights', 'report_format']:
                if param in report_params:
                    report_args[param] = report_params[param]
                    
        report_path = self.create_summary_report(**report_args)
        
        # Собираем все результаты
        pipeline_results = {
            'run_id': self.run_id,
            'run_dir': self.run_dir,
            'report_path': report_path,
            'strategy_profile': self.strategy_profile,
            'used_parameters': {
                'signal_params': signal_params,
                'standard_portfolio_params': standard_portfolio_params,
                'short_portfolio_params': short_portfolio_params,
                'combined_portfolio_params': combined_portfolio_params,
                'portfolio_controls': controls,
                'backtest_params': backtest_params,
                'optimization_params': optimization_params,
                'select_portfolio_params': select_portfolio_params,
                'report_params': report_params,
                'visualization_params': visualization_params
            },
            **self.results
        }
        
        self.logger.info(f"Выполнение пайплайна завершено. Результаты сохранены в {self.run_dir}")
        
        return pipeline_results
