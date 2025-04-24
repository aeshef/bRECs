import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

class Backtester:
    def __init__(self, input_file=None, portfolio_weights=None, start_date=None, end_date=None, log_level=logging.INFO):
        """
        Бэктестер для оценки эффективности стратегии
        
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
        """
        self.input_file = input_file
        self.portfolio_weights = portfolio_weights
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.returns_by_ticker = {}
        self.portfolio_returns = None
        self.metrics = None
        
        # Настройка логгера
        self.logger = logging.getLogger('backtester')
        self.logger.setLevel(log_level)
        
        # Создаем обработчик для записи в файл
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{log_dir}/backtester_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Создаем форматтер
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Добавляем обработчик к логгеру, если его еще нет
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            # Добавляем вывод в консоль
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def load_data(self, input_file=None):
        """Загрузка данных для бэктестирования"""
        if input_file:
            self.input_file = input_file
                
        if self.input_file:
            self.logger.info(f"Загрузка данных из {self.input_file}")
                
            try:
                self.df = pd.read_csv(self.input_file)
                    
                # Преобразование даты в индекс, если есть
                if 'date' in self.df.columns:
                    self.df['date'] = pd.to_datetime(self.df['date'])
                    self.df.set_index('date', inplace=True)
                        
                self.logger.info(f"Загружено {len(self.df)} строк данных")
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке данных: {e}")
                return None
        else:
            self.logger.error("Не указан источник данных")
            return None
                
        # Фильтрация по дате
        if self.start_date:
            self.df = self.df[self.df.index >= self.start_date]
        if self.end_date:
            self.df = self.df[self.df.index <= self.end_date]
                
        if self.df is not None and len(self.df) > 0:
            self.logger.info(f"Период бэктеста: {self.df.index.min().strftime('%Y-%m-%d')} - {self.df.index.max().strftime('%Y-%m-%d')}")
                
        return self.df

    
    def calculate_returns(self, df=None, signals_col='signal', price_col='close'):
        """
        Рассчитывает доходность на основе сигналов
        """
        if df is None:
            df = self.df
            
        if df is None:
            self.logger.error("Данные не загружены")
            return None
            
        # Проверка наличия нужных колонок
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
        
        # Стратегическая доходность: сигнал предыдущего дня * доходность текущего дня
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
        
        # Расчет доходности для каждого тикера
        if 'ticker' in self.df.columns:
            self.returns_by_ticker = {}
            
            for ticker, ticker_data in self.df.groupby('ticker'):
                self.returns_by_ticker[ticker] = self.calculate_returns(ticker_data.sort_index())
                
            self.logger.info(f"Рассчитаны доходности для {len(self.returns_by_ticker)} тикеров")
        else:
            # Если нет колонки ticker, считаем как для одного актива
            returns_df = self.calculate_returns(self.df)
            self.returns_by_ticker = {'SINGLE': returns_df}
            self.logger.info("Рассчитаны доходности для одного актива")
            
        return self.returns_by_ticker
    
    def calculate_portfolio_return(self, weights=None):
        """
        Рассчитывает доходность портфеля с заданными весами,
        поддерживает формат тикеров TICKER_LONG и TICKER_SHORT
        """
        if not self.returns_by_ticker:
            self.logger.error("Доходности по тикерам не рассчитаны")
            return None
            
        if weights is None:
            weights = self.portfolio_weights
            
        # Если веса не предоставлены, используем равные веса
        if weights is None:
            weights = {ticker: 1/len(self.returns_by_ticker) for ticker in self.returns_by_ticker.keys()}
            self.logger.info("Используются равные веса для всех активов")
        
        self.logger.info(f"Расчет доходности портфеля с {len(weights)} весами")
        
        # Создаем DataFrame для каждого тикера
        portfolio_returns = pd.DataFrame()
        
        # Сопоставляем веса с тикерами, поддерживая формат TICKER_LONG и TICKER_SHORT
        self.logger.info("Анализ данных доходностей:")
        for ticker, returns_df in self.returns_by_ticker.items():
            if 'strategy_return' in returns_df.columns:
                non_zero = (returns_df['strategy_return'] != 0).sum()
                total = len(returns_df)
                max_val = returns_df['strategy_return'].abs().max()
                self.logger.info(f"Тикер {ticker}: {non_zero}/{total} ненулевых точек, макс. значение: {max_val:.6f}")
                
            # Проверяем различные форматы тикеров в весах
            if ticker in weights and ticker != 'RISK_FREE':
                # Стандартный формат
                portfolio_returns[ticker] = returns_df['strategy_return'] * weights[ticker]
                self.logger.info(f"Добавлен тикер {ticker} с весом {weights[ticker]:.4f}")
            elif f"{ticker}_LONG" in weights:
                # Формат LONG: обычная доходность
                portfolio_returns[ticker] = returns_df['strategy_return'] * weights[f"{ticker}_LONG"]
                self.logger.info(f"Добавлен тикер {ticker} (LONG) с весом {weights[f'{ticker}_LONG']:.4f}")
            elif f"{ticker}_SHORT" in weights:
                # Формат SHORT: инвертированная доходность
                portfolio_returns[ticker] = -returns_df['strategy_return'] * weights[f"{ticker}_SHORT"]
                self.logger.info(f"Добавлен тикер {ticker} (SHORT) с весом {weights[f'{ticker}_SHORT']:.4f}")
        
        # Проверяем, что у нас есть хотя бы один тикер в портфеле
        if portfolio_returns.empty:
            self.logger.warning("Нет совпадений между тикерами в весах и данными")
            # Детальная отладка проблем сопоставления
            self.logger.info(f"Доступные тикеры в данных: {list(self.returns_by_ticker.keys())}")
            weight_tickers = [k for k in weights.keys() if k != 'RISK_FREE']
            self.logger.info(f"Тикеры в весах: {weight_tickers}")
            
            # Создаем пустой DataFrame для дальнейших расчётов
            dates = next(iter(self.returns_by_ticker.values())).index
            portfolio_returns = pd.DataFrame(index=dates)
            portfolio_returns['portfolio_return'] = 0.0
        else:
            # Рассчитываем общую доходность портфеля
            portfolio_returns['portfolio_return'] = portfolio_returns.sum(axis=1)
        
        # Если есть безрисковая часть, добавляем ее
        if 'RISK_FREE' in weights and weights['RISK_FREE'] > 0:
            rf_weight = weights['RISK_FREE']
            # Предполагаем безрисковую ставку в годовом исчислении, переводим в дневную
            risk_free_rate = 0.075  # Например, 7.5%
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
            
            self.logger.info(f"Добавлена безрисковая часть с весом {rf_weight*100:.1f}% и ставкой {risk_free_rate*100:.2f}%")
            
            # Пропорционально уменьшаем вес портфеля с акциями
            if 'portfolio_return' in portfolio_returns.columns:
                portfolio_returns['portfolio_return'] = portfolio_returns['portfolio_return'] * (1 - rf_weight)
            
            # Добавляем безрисковую часть
            portfolio_returns['risk_free_return'] = daily_rf_rate
            
            # Обновляем общую доходность
            if 'portfolio_return' in portfolio_returns.columns:
                portfolio_returns['portfolio_return'] = portfolio_returns['portfolio_return'] + daily_rf_rate * rf_weight
            else:
                portfolio_returns['portfolio_return'] = daily_rf_rate * rf_weight
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns

    
    def calculate_performance_metrics(self, returns=None, risk_free_rate=0):
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
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Максимальная просадка
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Винрейт (доля доходных дней)
        win_rate = (daily_returns > 0).mean()
        
        metrics = {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        self.metrics = metrics
        
        self.logger.info(f"Кумулятивная доходность: {cumulative_return*100:.2f}%, " + 
                         f"Годовая доходность: {annual_return*100:.2f}%, " + 
                         f"Шарп: {sharpe_ratio:.2f}, " + 
                         f"Макс. просадка: {max_drawdown*100:.2f}%")
        
        return metrics
    
    def visualize_results(self, output_dir=None):
        """
        Визуализирует результаты бэктеста
        """
        if self.portfolio_returns is None or self.metrics is None:
            self.logger.error("Результаты бэктеста не рассчитаны")
            return None
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.logger.info("Создание визуализации результатов бэктеста")
        
        # График кумулятивной доходности
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
        
        # График просадок
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
        
        # Сохранение метрик в файл
        if output_dir:
            with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
                f.write(f"Период: {self.portfolio_returns.index.min().strftime('%Y-%m-%d')} - {self.portfolio_returns.index.max().strftime('%Y-%m-%d')}\n\n")
                f.write(f"Кумулятивная доходность: {self.metrics['cumulative_return']*100:.2f}%\n")
                f.write(f"Годовая доходность: {self.metrics['annual_return']*100:.2f}%\n")
                f.write(f"Годовая волатильность: {self.metrics['annual_volatility']*100:.2f}%\n")
                f.write(f"Коэффициент Шарпа: {self.metrics['sharpe_ratio']:.2f}\n")
                f.write(f"Максимальная просадка: {self.metrics['max_drawdown']*100:.2f}%\n")
                f.write(f"Винрейт: {self.metrics['win_rate']*100:.2f}%\n")
            
            # Сохранение портфельных доходностей
            self.portfolio_returns.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))
            
            self.logger.info(f"Метрики и данные сохранены в {output_dir}")
    
    def run_pipeline(self, input_file=None, portfolio_weights=None, 
               start_date=None, end_date=None, output_dir="/Users/aeshef/Documents/GitHub/kursach/data/portfolio/results", 
               risk_free_rate=7.5):
        """
        Запускает полный пайплайн бэктестирования
        
        Parameters:
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
            
        Returns:
        --------
        dict с результатами бэктеста
        """
        self.logger.info("Запуск пайплайна бэктестирования")

        # Устанавливаем путь к файлу по умолчанию, если не указан
        if input_file is None:
            input_file = "/Users/aeshef/Documents/GitHub/kursach/data/signals.csv"
        
        # Обновление параметров, если указаны
        if portfolio_weights is not None:
            self.portfolio_weights = portfolio_weights
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
            
        # Загрузка данных (только из файла, убираем df)
        self.load_data(input_file)
        
        if self.df is None or len(self.df) == 0:
            self.logger.error("Не удалось загрузить данные для бэктеста")
            return None
            
        # Расчет доходностей по тикерам
        self.calculate_returns_by_ticker()
        
        if not self.returns_by_ticker:
            self.logger.error("Не удалось рассчитать доходности по тикерам")
            return None
            
        # Расчет доходности портфеля
        self.calculate_portfolio_return()
        
        if self.portfolio_returns is None:
            self.logger.error("Не удалось рассчитать доходность портфеля")
            return None
            
        # Расчет метрик эффективности
        self.calculate_performance_metrics(risk_free_rate=risk_free_rate)
        
        if self.metrics is None:
            self.logger.error("Не удалось рассчитать метрики эффективности")
            return None
            
        # Визуализация результатов
        if output_dir:
            self.visualize_results(output_dir)
            
        # Возвращаем результаты
        results = {
            'metrics': self.metrics,
            'returns': self.portfolio_returns,
            'cumulative_returns': (1 + self.portfolio_returns['portfolio_return'].fillna(0)).cumprod()
        }
        
        return results
