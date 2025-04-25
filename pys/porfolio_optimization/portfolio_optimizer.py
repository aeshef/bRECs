import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import logging
from datetime import datetime
import sys

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class PortfolioOptimizer(BaseLogger):
    def __init__(self, input_file=None, max_weight=0.2, risk_free_rate=0.06, min_rf_allocation=0.25, 
                max_rf_allocation=0.35, log_level=logging.INFO):
        """
        Оптимизатор портфеля с использованием модели Марковица
        
        Parameters:
        -----------
        input_file : str, optional
            Путь к файлу с данными для оптимизации
        risk_free_rate : float
            Безрисковая ставка (в десятичном формате, например 0.06 = 6%)
        min_rf_allocation : float
            Минимальная доля безрисковых активов в портфеле
        max_rf_allocation : float
            Максимальная доля безрисковых активов в портфеле
        """
        super().__init__('PortfolioOptimizer')
        self.input_file = input_file
        self.risk_free_rate = risk_free_rate
        self.min_rf_allocation = min_rf_allocation
        self.max_rf_allocation = max_rf_allocation
        self.df = None
        self.returns = None
        self.optimal_weights = None
        self.portfolio_performance = None
        self.max_weight = max_weight
        
        # # Настройка логгера
        # self.logger = logging.getLogger('portfolio_optimizer')
        # self.logger.setLevel(log_level)
        
        # # Создаем обработчик для записи в файл
        # log_dir = 'logs'
        # os.makedirs(log_dir, exist_ok=True)
        # file_handler = logging.FileHandler(f'{log_dir}/portfolio_optimizer_{datetime.now().strftime("%Y%m%d")}.log')
        
        # # Создаем форматтер
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # file_handler.setFormatter(formatter)
        
        # # Добавляем обработчик к логгеру, если его еще нет
        # if not self.logger.handlers:
        #     self.logger.addHandler(file_handler)
        #     # Добавляем вывод в консоль
        #     console_handler = logging.StreamHandler()
        #     console_handler.setFormatter(formatter)
        #     self.logger.addHandler(console_handler)
    
    def load_data(self, input_file=None, df=None):
        """Загрузка данных для оптимизации"""
        if df is not None:
            self.df = df
            self.logger.info(f"Использован предоставленный DataFrame с {len(df)} строками")
            return self.df
            
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
                return self.df
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке данных: {e}")
                return None
        else:
            self.logger.error("Не указан источник данных")
            return None
    
    def prepare_returns(self):
        """Подготовка доходностей для оптимизации"""
        if self.df is None:
            self.logger.error("Данные не загружены")
            return None
            
        self.logger.info("Подготовка данных доходностей для оптимизации")
        
        try:
            # Фильтрация по сигналам и shortlist, если есть
            filtered_df = self.df
            
            if 'in_shortlist' in self.df.columns and 'signal' in self.df.columns:
                filtered_df = self.df[(self.df['in_shortlist'] == True) & (self.df['signal'] >= 0)]
                self.logger.info(f"Отфильтровано {len(filtered_df)} строк по shortlist и сигналам")
                
            # Если есть колонка ticker, группируем по тикерам
            if 'ticker' in filtered_df.columns:
                # Расчет доходностей для каждого тикера отдельно
                returns_dict = {}
                
                for ticker, ticker_data in filtered_df.groupby('ticker'):
                    ticker_data = ticker_data.sort_index()
                    if 'close' in ticker_data.columns:
                        returns_dict[ticker] = ticker_data['close'].pct_change().dropna()
                        
                # Объединение всех доходностей в одну таблицу
                self.returns = pd.DataFrame(returns_dict)
                self.logger.info(f"Рассчитаны доходности для {len(returns_dict)} тикеров")
            else:
                # Если нет колонки ticker, рассчитываем по close для всего DataFrame
                if 'close' in filtered_df.columns:
                    self.returns = filtered_df['close'].pct_change().dropna()
                    self.logger.info("Рассчитаны доходности на основе колонки 'close'")
                else:
                    self.logger.error("Не найдена колонка 'close' для расчета доходностей")
                    return None
                
            # Проверка наличия данных
            if self.returns is None or (isinstance(self.returns, pd.DataFrame) and self.returns.empty):
                self.logger.error("Не удалось рассчитать доходности")
                return None
                
            return self.returns
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке доходностей: {e}")
            return None
    
    def calculate_portfolio_performance(self, weights, returns=None):
        """
        Рассчитывает доходность и риск портфеля
        """
        if returns is None:
            returns = self.returns
            
        if returns is None:
            self.logger.error("Доходности не рассчитаны")
            return (0, 0)
            
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Годовая доходность
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Годовая волатильность
        
        return portfolio_return, portfolio_vol
    
    def negative_sharpe_ratio(self, weights, returns, risk_free_rate):
        """
        Отрицательный коэффициент Шарпа (для минимизации)
        """
        p_ret, p_vol = self.calculate_portfolio_performance(weights, returns)
        return -(p_ret - risk_free_rate) / p_vol
    
    def optimize_portfolio(self, returns=None, risk_free_rate=None, constrained=True, bounds=None):
        """
        Оптимизирует портфель, максимизируя коэффициент Шарпа
        """
        if returns is None:
            returns = self.returns
            
        if returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
            
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if bounds is None and constrained:
            bounds = tuple((0, self.max_weight) for _ in range(n))
            
        self.logger.info(f"Запуск оптимизации портфеля для {returns.shape[1]} активов")
        
        n = len(returns.columns)
        
        # Ограничения: сумма весов = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Границы весов: от 0 до 1 для каждого актива
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n))
        
        # Начальные веса: равные для всех активов
        init_weights = np.array([1/n] * n)
        
        try:
            # Оптимизация
            results = minimize(self.negative_sharpe_ratio, 
                               init_weights, 
                               args=(returns, risk_free_rate),
                               method='SLSQP', 
                               bounds=bounds if constrained else None,
                               constraints=constraints)
            
            if results['success']:
                self.optimal_weights = results['x']
                self.logger.info("Оптимизация успешно завершена")
            else:
                self.logger.warning(f"Оптимизация не сошлась: {results['message']}")
                self.optimal_weights = init_weights  # Используем равные веса как запасной вариант
                
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при оптимизации портфеля: {e}")
            return None
    
    def calculate_final_portfolio(self, rf_allocation=None):
        """
        Рассчитывает итоговый портфель с учетом безрисковой части
        """
        if self.optimal_weights is None or self.returns is None:
            self.logger.error("Оптимальные веса не рассчитаны")
            return None
            
        # Если доля безрисковых активов не указана, берем среднее из диапазона
        if rf_allocation is None:
            rf_allocation = (self.min_rf_allocation + self.max_rf_allocation) / 2
            
        self.logger.info(f"Расчет итогового портфеля с долей безрисковых активов {rf_allocation*100:.1f}%")
        
        # Создание словаря с весами
        tickers = self.returns.columns
        risky_weights = dict(zip(tickers, self.optimal_weights))
        
        # Пересчет весов с учетом безрисковой части
        final_weights = {ticker: weight * (1 - rf_allocation) for ticker, weight in risky_weights.items()}
        
        # Добавление безрисковой части в веса
        final_weights['RISK_FREE'] = rf_allocation
        
        # Расчет ожидаемой доходности и волатильности оптимального портфеля
        expected_return, expected_volatility = self.calculate_portfolio_performance(self.optimal_weights)
        
        # Расчет общей ожидаемой доходности с учетом безрисковой части
        total_expected_return = expected_return * (1 - rf_allocation) + self.risk_free_rate * rf_allocation
        total_expected_volatility = expected_volatility * (1 - rf_allocation)
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        # Сохранение результатов
        self.portfolio_performance = {
            'weights': final_weights,
            'expected_return': total_expected_return,
            'expected_volatility': total_expected_volatility,
            'sharpe_ratio': sharpe_ratio,
            'risk_free_rate': self.risk_free_rate,
            'rf_allocation': rf_allocation
        }
        
        self.logger.info(f"Ожидаемая доходность портфеля: {total_expected_return*100:.2f}%, " + 
                         f"волатильность: {total_expected_volatility*100:.2f}%, " + 
                         f"Шарп: {sharpe_ratio:.2f}")
        
        return self.portfolio_performance
    
    def generate_efficient_frontier(self, num_portfolios=1000):
        """
        Генерирует множество случайных портфелей и находит эффективную границу
        
        Parameters:
        -----------
        num_portfolios : int
            Количество случайных портфелей для генерации
            
        Returns:
        --------
        dict
            Словарь с данными о случайных портфелях и эффективной границе
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
            
        self.logger.info(f"Генерация {num_portfolios} случайных портфелей для построения эффективной границы")
        
        n_assets = self.returns.shape[1]
        
        # Создаем массивы для хранения результатов
        all_weights = np.zeros((num_portfolios, n_assets))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)
        
        # Генерация случайных портфелей
        for i in range(num_portfolios):
            # Генерируем случайные веса
            weights = np.random.random(n_assets)
            # Нормализуем веса, чтобы их сумма равнялась 1
            weights = weights / np.sum(weights)
            
            # Сохраняем веса
            all_weights[i, :] = weights
            
            # Рассчитываем ожидаемую доходность и волатильность
            portfolio_return, portfolio_vol = self.calculate_portfolio_performance(weights)
            
            # Сохраняем результаты
            ret_arr[i] = portfolio_return
            vol_arr[i] = portfolio_vol
            # Рассчитываем коэффициент Шарпа
            sharpe_arr[i] = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Находим оптимальный портфель по Шарпу
        max_sharpe_idx = sharpe_arr.argmax()
        max_sharpe_return = ret_arr[max_sharpe_idx]
        max_sharpe_vol = vol_arr[max_sharpe_idx]
        max_sharpe_weights = all_weights[max_sharpe_idx, :]
        
        # Находим портфель с минимальной волатильностью
        min_vol_idx = vol_arr.argmin()
        min_vol_return = ret_arr[min_vol_idx]
        min_vol_vol = vol_arr[min_vol_idx]
        min_vol_weights = all_weights[min_vol_idx, :]
        
        # Собираем все данные в словарь
        frontier_data = {
            'returns': ret_arr,
            'volatilities': vol_arr,
            'sharpe_ratios': sharpe_arr,
            'weights': all_weights,
            'max_sharpe_idx': max_sharpe_idx,
            'max_sharpe_return': max_sharpe_return,
            'max_sharpe_vol': max_sharpe_vol,
            'max_sharpe_weights': max_sharpe_weights,
            'min_vol_idx': min_vol_idx,
            'min_vol_return': min_vol_return,
            'min_vol_vol': min_vol_vol,
            'min_vol_weights': min_vol_weights
        }
        
        return frontier_data

    def plot_efficient_frontier(self, frontier_data=None, output_dir=None):
        """
        Визуализирует эффективную границу и множество портфелей
        
        Parameters:
        -----------
        frontier_data : dict, optional
            Данные эффективной границы (если None, будут сгенерированы)
        output_dir : str, optional
            Директория для сохранения результатов
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
            
        if frontier_data is None:
            frontier_data = self.generate_efficient_frontier()
            
        if frontier_data is None:
            self.logger.error("Не удалось сгенерировать данные для эффективной границы")
            return None
        
        self.logger.info("Создание графика эффективной границы")
        
        # Создаем новую фигуру
        plt.figure(figsize=(12, 8))
        
        # Облако случайных портфелей
        scatter = plt.scatter(
            frontier_data['volatilities'], 
            frontier_data['returns'], 
            c=frontier_data['sharpe_ratios'],
            cmap='viridis', 
            alpha=0.5,
            s=10
        )
        
        # Отмечаем точки с максимальным Шарпом и минимальной волатильностью
        plt.scatter(
            frontier_data['max_sharpe_vol'], 
            frontier_data['max_sharpe_return'],
            c='red', 
            marker='*', 
            s=300, 
            label='Максимальный коэффициент Шарпа'
        )
        
        plt.scatter(
            frontier_data['min_vol_vol'], 
            frontier_data['min_vol_return'],
            c='green', 
            marker='o', 
            s=200, 
            label='Минимальная волатильность'
        )
        
        # Отображаем оптимальный портфель, если он был рассчитан
        if hasattr(self, 'portfolio_performance') and self.portfolio_performance is not None:
            plt.scatter(
                self.portfolio_performance['expected_volatility'],
                self.portfolio_performance['expected_return'],
                c='blue', 
                marker='d', 
                s=200, 
                label='Оптимальный портфель'
            )
        
        # Добавляем линию CML (Capital Market Line)
        if self.risk_free_rate is not None:
            # CML проходит от безрисковой ставки через портфель с максимальным Шарпом
            x_cml = np.linspace(0, max(frontier_data['volatilities']) * 1.2, 100)
            max_sharpe_slope = (frontier_data['max_sharpe_return'] - self.risk_free_rate) / frontier_data['max_sharpe_vol']
            y_cml = self.risk_free_rate + max_sharpe_slope * x_cml
            
            plt.plot(
                x_cml, 
                y_cml, 
                'g--', 
                linewidth=2, 
                label='Линия рынка капитала (CML)'
            )
            
            # Отмечаем безрисковую ставку
            plt.scatter(
                0, 
                self.risk_free_rate,
                c='black', 
                marker='o', 
                s=100, 
                label='Безрисковая ставка'
            )
        
        # Добавляем цветовую шкалу для коэффициента Шарпа
        cbar = plt.colorbar(scatter)
        cbar.set_label('Коэффициент Шарпа')
        
        # Добавляем подписи и заголовок
        plt.title('Эффективная граница и облако портфелей')
        plt.xlabel('Ожидаемая волатильность (годовая)')
        plt.ylabel('Ожидаемая доходность (годовая)')
        plt.legend(loc='best')
        
        # Добавляем сетку для лучшей читаемости
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график, если указана директория
        if output_dir:
            frontier_path = os.path.join(output_dir, 'efficient_frontier.png')
            plt.savefig(frontier_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"График эффективной границы сохранен в {frontier_path}")
        else:
            plt.show()
        
        # Обязательно закрываем фигуру после создания
        plt.close()
        
        return frontier_data


    def plot_enhanced_efficient_frontier(self, frontier_data=None, output_dir=None):
        """
        Создает улучшенную визуализацию эффективной границы и оптимальных портфелей
        с выделением границы эффективных портфелей и точки касания с CML
        
        Parameters:
        -----------
        frontier_data : dict, optional
            Данные эффективной границы (если None, будут сгенерированы)
        output_dir : str, optional
            Директория для сохранения результатов
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
            
        if frontier_data is None:
            frontier_data = self.generate_efficient_frontier(num_portfolios=5000)
            
        if frontier_data is None:
            self.logger.error("Не удалось сгенерировать данные для эффективной границы")
            return None
        
        self.logger.info("Создание улучшенного графика эффективной границы")
        
        # Создаем фигуру с двумя областями - основной график и распределение портфеля
        fig = plt.figure(figsize=(16, 10))
        
        # Основная область для эффективной границы
        ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=5, rowspan=3)
        
        # Область для отображения состава портфеля
        ax2 = plt.subplot2grid((5, 5), (3, 0), colspan=5, rowspan=2)
        
        # Извлекаем данные
        vol_arr = frontier_data['volatilities']
        ret_arr = frontier_data['returns']
        sharpe_arr = frontier_data['sharpe_ratios']
        all_weights = frontier_data['weights']
        
        # Создаем датафрейм для удобства работы
        df = pd.DataFrame({
            'Volatility': vol_arr,
            'Returns': ret_arr,
            'Sharpe': sharpe_arr
        })
        
        # Сортируем по волатильности для построения границы
        df = df.sort_values('Volatility')
        
        # Построение эффективной границы
        # Находим портфели, которые образуют выпуклую границу
        # Используем скользящее окно для нахождения максимальной доходности при заданной волатильности
        window_size = 100
        vol_windows = []
        ret_windows = []
        
        # Делим диапазон волатильности на равные интервалы
        vol_min = df['Volatility'].min()
        vol_max = df['Volatility'].max()
        
        vol_steps = np.linspace(vol_min, vol_max, 50)
        
        for i in range(len(vol_steps)-1):
            vol_lower = vol_steps[i]
            vol_upper = vol_steps[i+1]
            
            # Выбираем портфели в этом диапазоне волатильности
            window_portfolios = df[(df['Volatility'] >= vol_lower) & (df['Volatility'] <= vol_upper)]
            
            if not window_portfolios.empty:
                # Выбираем портфель с максимальной доходностью в этом окне
                max_return_idx = window_portfolios['Returns'].idxmax()
                vol_windows.append(df.loc[max_return_idx, 'Volatility'])
                ret_windows.append(df.loc[max_return_idx, 'Returns'])
        
        # Для построения сглаженной кривой используем LOWESS
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(ret_windows, vol_windows, frac=0.25)
            vol_smooth, ret_smooth = smoothed.T
        except ImportError:
            # Если statsmodels не установлен, используем простую сортировку
            vol_smooth = np.array(vol_windows)
            ret_smooth = np.array(ret_windows)
            sort_idx = np.argsort(vol_smooth)
            vol_smooth = vol_smooth[sort_idx]
            ret_smooth = ret_smooth[sort_idx]
        
        # Рисуем облако точек
        scatter = ax1.scatter(
            vol_arr, 
            ret_arr, 
            c=sharpe_arr,
            cmap='viridis', 
            s=5,
            alpha=0.2
        )
        
        # Рисуем эффективную границу
        ax1.plot(vol_smooth, ret_smooth, 'r-', linewidth=3, label='Эффективная граница')
        
        # Вычисляем и рисуем CML - линию рынка капитала
        max_sharpe_vol = frontier_data['max_sharpe_vol']
        max_sharpe_return = frontier_data['max_sharpe_return']
        
        # Создаем линию CML от безрисковой ставки до точки касания и дальше
        x_cml = np.linspace(0, max(vol_arr) * 1.2, 100)
        slope = (max_sharpe_return - self.risk_free_rate) / max_sharpe_vol
        y_cml = self.risk_free_rate + slope * x_cml
        
        ax1.plot(
            x_cml, 
            y_cml, 
            'g--', 
            linewidth=2.5, 
            label='Линия рынка капитала (CML)'
        )
        
        # Безрисковая ставка
        ax1.scatter(
            0, 
            self.risk_free_rate,
            c='black', 
            marker='o', 
            s=150, 
            label='Безрисковая ставка'
        )
        
        # Находим и отмечаем точку касания (максимальный Шарп)
        ax1.scatter(
            max_sharpe_vol, 
            max_sharpe_return,
            c='gold', 
            marker='*', 
            s=400, 
            edgecolor='black',
            linewidth=1.5,
            label='Касательный портфель (макс. Шарп)'
        )
        
        # Находим и отмечаем портфель с минимальной волатильностью
        min_vol_vol = frontier_data['min_vol_vol']
        min_vol_return = frontier_data['min_vol_return']
        
        ax1.scatter(
            min_vol_vol, 
            min_vol_return,
            c='lightgreen', 
            marker='o', 
            s=200, 
            edgecolor='black',
            linewidth=1.5,
            label='Портфель минимального риска'
        )
        
        # Если есть оптимальный портфель, добавляем его
        if hasattr(self, 'portfolio_performance') and self.portfolio_performance is not None:
            ax1.scatter(
                self.portfolio_performance['expected_volatility'],
                self.portfolio_performance['expected_return'],
                c='blue', 
                marker='d', 
                s=250, 
                edgecolor='black',
                linewidth=1.5,
                label='Оптимальный портфель'
            )
        
        # Настройка основного графика
        ax1.set_title('Эффективная граница и оптимальные портфели', fontsize=16)
        ax1.set_xlabel('Ожидаемая волатильность (годовая)', fontsize=12)
        ax1.set_ylabel('Ожидаемая доходность (годовая)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        
        # Добавляем аннотацию с числовыми характеристиками касательного портфеля
        info_text = (
            f"Касательный портфель:\n"
            f"Доходность: {max_sharpe_return*100:.2f}%\n"
            f"Волатильность: {max_sharpe_vol*100:.2f}%\n"
            f"Коэффициент Шарпа: {sharpe_arr.max():.2f}"
        )
        
        ax1.annotate(
            info_text,
            xy=(max_sharpe_vol, max_sharpe_return),
            xytext=(max_sharpe_vol+0.02, max_sharpe_return+0.02),
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            fontsize=9
        )
        
        # На нижнем графике показываем состав касательного портфеля
        max_sharpe_idx = frontier_data['max_sharpe_idx']
        max_sharpe_weights = all_weights[max_sharpe_idx]
        
        # Сортируем веса по убыванию для лучшей визуализации
        asset_names = list(self.returns.columns)
        weights_dict = {asset: weight for asset, weight in zip(asset_names, max_sharpe_weights)}
        sorted_weights = {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # Фильтруем малые веса для лучшей визуализации
        significant_weights = {k: v for k, v in sorted_weights.items() if v > 0.01}
        
        if len(significant_weights) < len(sorted_weights):
            other_weight = sum(v for k, v in sorted_weights.items() if v <= 0.01)
            if other_weight > 0:
                significant_weights['Другие'] = other_weight
        
        # Горизонтальная столбчатая диаграмма для весов
        bars = ax2.barh(
            list(significant_weights.keys()), 
            list(significant_weights.values()),
            color=plt.cm.viridis(np.linspace(0, 0.8, len(significant_weights))),
            edgecolor='white',
            linewidth=0.7
        )
        
        # Добавляем проценты к столбцам
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{width*100:.1f}%',
                va='center'
            )
        
        ax2.set_title('Состав касательного портфеля', fontsize=14)
        ax2.set_xlabel('Доля в портфеле', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, max(significant_weights.values()) * 1.15)  # Добавляем место для подписей
        
        # Добавляем цветовую шкалу для коэффициента Шарпа
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Коэффициент Шарпа')
        
        plt.tight_layout()
        
        # Сохраняем график, если указана директория
        if output_dir:
            frontier_path = os.path.join(output_dir, 'enhanced_efficient_frontier.png')
            plt.savefig(frontier_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Улучшенный график эффективной границы сохранен в {frontier_path}")
        else:
            plt.show()
        
        # Обязательно закрываем фигуру после создания
        plt.close()
        
        return frontier_data

    
    def visualize_portfolio(self, output_dir=None):
        """
        Визуализирует итоговый портфель с помощью круговой диаграммы и эффективной границы
        """
        if self.portfolio_performance is None:
            self.logger.error("Нет данных для визуализации")
            return None
                
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
                
        self.logger.info("Создание визуализации портфеля")
        
        # График весов в виде круговой диаграммы (пай-чарт)
        plt.figure(figsize=(12, 8))
        weights = self.portfolio_performance['weights']

        # Сортируем веса по убыванию
        sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
        
        # Фильтруем веса > 0.5% для лучшей визуализации
        significant_weights = {ticker: weight for ticker, weight in sorted_weights.items() if weight > 0.005}
        
        # Если есть малозначительные веса, группируем их
        other_weight = 1.0 - sum(significant_weights.values())
        if other_weight > 0:
            significant_weights['Другие'] = other_weight
        
        # Создаем пай-чарт
        plt.pie(
            significant_weights.values(), 
            labels=significant_weights.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            explode=[0.05] * len(significant_weights)  # Небольшое смещение всех сегментов
        )
        plt.axis('equal')  # Обеспечивает круглую форму
        plt.title('Оптимальные веса портфеля')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'portfolio_weights_pie.png'))
            self.logger.info(f"График весов сохранен в {output_dir}/portfolio_weights_pie.png")
        else:
            plt.show()
            
        # Обязательно закрываем фигуру после создания
        plt.close()
        
        # Создание улучшенного графика эффективной границы
        try:
            self.plot_enhanced_efficient_frontier(output_dir=output_dir)
            self.logger.info("Улучшенный график эффективной границы успешно создан")
        except Exception as e:
            self.logger.error(f"Ошибка при создании улучшенного графика эффективной границы: {e}")
            # В случае ошибки пробуем создать обычный график
            try:
                self.plot_efficient_frontier(output_dir=output_dir)
                self.logger.info("Стандартный график эффективной границы успешно создан")
            except Exception as e2:
                self.logger.error(f"Ошибка при создании стандартного графика эффективной границы: {e2}")
        
        if output_dir:
            weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
            weights_df.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'), index=False)
            
            with open(os.path.join(output_dir, 'portfolio_summary.txt'), 'w') as f:
                f.write(f"Ожидаемая годовая доходность: {self.portfolio_performance['expected_return']*100:.2f}%\n")
                f.write(f"Ожидаемая годовая волатильность: {self.portfolio_performance['expected_volatility']*100:.2f}%\n")
                f.write(f"Коэффициент Шарпа: {self.portfolio_performance['sharpe_ratio']:.2f}\n")
                f.write(f"Безрисковая ставка: {self.portfolio_performance['risk_free_rate']*100:.2f}%\n")
                f.write(f"Доля безрисковых активов: {self.portfolio_performance['rf_allocation']*100:.2f}%\n")
                    
            self.logger.info(f"Результаты сохранены в {output_dir}")



    
    def run_pipeline(self, input_file=None, output_dir=f"{BASE_PATH}/data",
              risk_free_rate=None, min_rf_allocation=None, max_rf_allocation=None, max_weight=None):
        """
        Запускает полный пайплайн оптимизации портфеля
        
        Parameters:
        -----------
        input_file : str, optional
            Путь к файлу с данными
        output_dir : str, optional
            Директория для сохранения результатов
        risk_free_rate : float, optional
            Безрисковая ставка
        min_rf_allocation : float, optional
            Минимальная доля безрисковых активов
        max_rf_allocation : float, optional
            Максимальная доля безрисковых активов
        max_weight : float, optional
            Максимальный вес одной бумаги в портфеле (для диверсификации)
        
        Returns:
        --------
        dict
            Словарь с результатами оптимизации
        """
        # Закрываем все предыдущие фигуры для предотвращения утечек памяти
        plt.close('all')
        self.logger.info("Запуск пайплайна оптимизации портфеля")
        
        # Обновление параметров, если указаны
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        if min_rf_allocation is not None:
            self.min_rf_allocation = min_rf_allocation
        if max_rf_allocation is not None:
            self.max_rf_allocation = max_rf_allocation
        if max_weight is not None:
            self.max_weight = max_weight
            self.logger.info(f"Максимальный вес одной бумаги: {self.max_weight}")
            
        # Использование input_file если не указан, берем значение по умолчанию
        if input_file is None:
            input_file = f"{BASE_PATH}/data/signals.csv"
            
        # Загрузка данных
        self.load_data(input_file)
        
        if self.df is None:
            self.logger.error("Не удалось загрузить данные")
            return None
            
        # Подготовка доходностей
        self.prepare_returns()
        
        if self.returns is None or (isinstance(self.returns, pd.DataFrame) and self.returns.empty):
            self.logger.error("Не удалось рассчитать доходности")
            return None
            
        # Оптимизация портфеля с учетом max_weight
        self.optimize_portfolio(constrained=True, 
                            bounds=tuple((0, self.max_weight) for _ in range(len(self.returns.columns))))
        
        if self.optimal_weights is None:
            self.logger.error("Не удалось оптимизировать портфель")
            return None
            
        # Расчет итогового портфеля с безрисковой частью
        portfolio = self.calculate_final_portfolio()
        
        # Визуализация результатов
        if output_dir:
            # Добавляем подпапку "portfolio" к указанной директории
            output_dir = os.path.join(output_dir, 'portfolio')
            # Убедимся, что все графики закрываются после создания
            try:
                self.visualize_portfolio(output_dir)
            except Exception as e:
                self.logger.error(f"Ошибка при визуализации: {e}")
            finally:
                # Дополнительно закрываем все фигуры после визуализации
                plt.close('all')
                
        return portfolio
