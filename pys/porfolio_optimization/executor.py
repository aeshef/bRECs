import os
import datetime
import shutil
import json
import pandas as pd
import logging
from pys.utils.logger import BaseLogger

class PipelineExecutor(BaseLogger):
    """
    Выполняет и организует финальные этапы пайплайна инвестиционной стратегии.
    
    Класс выполняет:
    - Генерацию сигналов
    - Оптимизацию стандартного портфеля
    - Создание портфеля с короткими позициями
    - Создание отчета о запуске
    
    Все результаты сохраняются в отдельной директории для каждого запуска.
    """
    
    def __init__(self, base_path, bond_results=None, name=None):
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
        log_level : int, optional
            Уровень логирования
        """
        super().__init__('PipelineExecutor')
        
        # Создаем идентификатор запуска с временной меткой
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name_suffix = f"_{name}" if name else ""
        self.run_id = f"run_{self.timestamp}{name_suffix}"
        
        self.base_path = base_path
        self.bond_results = bond_results
        
        # Создаем директорию для текущего запуска
        self.run_dir = os.path.join(base_path, "data", "pipeline_runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Создаем поддиректории для разных этапов
        self.signals_dir = os.path.join(self.run_dir, "signals")
        self.portfolio_dir = os.path.join(self.run_dir, "portfolio")
        self.backtest_dir = os.path.join(self.run_dir, "backtest")
        self.shorts_dir = os.path.join(self.run_dir, "shorts_portfolio")
        
        for directory in [self.signals_dir, self.portfolio_dir, self.backtest_dir, self.shorts_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Создана структура директорий для запуска {self.run_id}")
        
        # Инициализируем параметры и результаты
        self.signal_params = {}
        self.portfolio_params = {}
        self.short_params = {}
        self.results = {
            'bond_portfolio': None,
            'signals': None,
            'standard_portfolio': None,
            'short_portfolio': None
        }
    
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
    
    def generate_signals(self, weight_tech=0.5, weight_sentiment=0.3, weight_fundamental=0.2):
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
            "weight_fundamental": weight_fundamental
        }
        
        self.logger.info(f"Запуск генерации сигналов с параметрами: {self.signal_params}")
        
        # Запускаем генерацию сигналов
        signals = run_pipeline_signal_generator(
            weight_tech=weight_tech,
            weight_sentiment=weight_sentiment,
            weight_fundamental=weight_fundamental,
            output_dir=self.signals_dir  # Указываем директорию для визуализаций
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
    
    def optimize_standard_portfolio(self, tickers_list, risk_free_rate=0.1, min_rf_allocation=0.3, 
                                    max_rf_allocation=0.5, max_weight=0.15):
        """
        Запускает оптимизацию стандартного портфеля.
        
        Parameters:
        -----------
        tickers_list : list
            Список тикеров
        risk_free_rate : float
            Безрисковая ставка
        min_rf_allocation, max_rf_allocation : float
            Минимальная и максимальная доля безрисковых активов
        max_weight : float
            Максимальный вес одного актива
        
        Returns:
        --------
        dict
            Результаты оптимизации портфеля
        """
        from pys.porfolio_optimization.portfolio_optimizer import run_all_optimization_models
        
        self.portfolio_params = {
            "risk_free_rate": risk_free_rate,
            "min_rf_allocation": min_rf_allocation,
            "max_rf_allocation": max_rf_allocation,
            "max_weight": max_weight
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
                risk_free_portfolio_file=bond_portfolio_path
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
            
            # Сохраняем параметры оптимизации
            params_file = os.path.join(self.portfolio_dir, "portfolio_params.json")
            with open(params_file, 'w') as f:
                json.dump(self.portfolio_params, f, indent=4)
            
            self.results['standard_portfolio'] = portfolio_results
            self.logger.info(f"Портфель успешно оптимизирован и сохранен в {self.portfolio_dir}")
            
            return portfolio_results
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации портфеля: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_short_portfolio(self, risk_free_rate=0.075, train_period=('2024-01-01', '2024-12-31'), 
                           test_period=('2025-01-01', '2025-06-30'), best_params_file=None):
        """
        Создает портфель с короткими позициями.
        
        Parameters:
        -----------
        risk_free_rate : float
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
        self.short_params = {
            "risk_free_rate": risk_free_rate,
            "train_period": train_period,
            "test_period": test_period
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
                verify_with_honest_backtest=True,
                train_period=train_period,
                test_period=test_period
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

    
    def create_summary_report(self):
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
                f.write(f"**Идентификатор запуска:** {self.run_id}\n\n")
                
                # Параметры
                f.write("## Параметры\n\n")
                
                if self.signal_params:
                    f.write("### Параметры генерации сигналов\n")
                    for key, value in self.signal_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if self.portfolio_params:
                    f.write("### Параметры оптимизации портфеля\n")
                    for key, value in self.portfolio_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if self.short_params:
                    f.write("### Параметры портфеля с короткими позициями\n")
                    for key, value in self.short_params.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                # Результаты
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
                f.write("├── backtest/           # Результаты бэктестов\n")
                f.write("└── bond_portfolio.csv  # Портфель облигаций\n")
                f.write("```\n")
            
            self.logger.info(f"Итоговый отчет создан: {summary_path}")
            return summary_path
        except Exception as e:
            self.logger.error(f"Ошибка при создании итогового отчета: {e}")
            return None
        
    # Add this method to the PipelineExecutor class

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

    
    def run_pipeline(self, tickers_list, bond_results=None):
        """
        Запускает все этапы пайплайна последовательно.
        
        Parameters:
        -----------
        tickers_list : list
            Список тикеров
        bond_results : dict, optional
            Результаты выполнения run_bond_selection_with_kbd
        
        Returns:
        --------
        dict
            Результаты выполнения всех этапов пайплайна
        """
        self.logger.info(f"Запуск полного пайплайна с ID: {self.run_id}")
        
        # 1. Копируем портфель облигаций
        self.copy_bond_portfolio(bond_results)
        
        # 2. Генерируем сигналы
        self.generate_signals()
        
        # 3. Оптимизируем стандартный портфель
        self.optimize_standard_portfolio(tickers_list)
        
        # 4. Создаем портфель с короткими позициями
        self.create_short_portfolio()
        
        # 5. Создаем итоговый отчет
        report_path = self.create_summary_report()
        
        # Собираем все результаты
        pipeline_results = {
            'run_id': self.run_id,
            'run_dir': self.run_dir,
            'report_path': report_path,
            **self.results
        }
        
        self.logger.info(f"Выполнение пайплайна завершено. Результаты сохранены в {self.run_dir}")
        
        return pipeline_results
