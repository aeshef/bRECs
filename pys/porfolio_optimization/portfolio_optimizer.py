import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import logging
from datetime import datetime
import sys
import json

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class PortfolioOptimizer(BaseLogger):
    def __init__(self, input_file=None, max_weight=0.2, risk_free_rate=0.06, min_rf_allocation=0.25, 
                max_rf_allocation=0.35, optimization='markowitz', tau=0.05, 
                views=None, view_confidences=None, market_caps=None, log_level=logging.INFO, risk_free_portfolio_file=None,
                include_short_selling=False):
        """
        Оптимизатор портфеля с использованием модели Марковица или Блэка-Литермана
        
        Parameters:
        -----------
        ...
        market_caps : dict, optional
            Рыночные капитализации активов для модели Блэка-Литермана
        ...
        """
        import pandas as pd 
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
        
        self.risk_free_portfolio_file = risk_free_portfolio_file
        self.risk_free_portfolio = None  # Будет хранить DataFrame с деталями облигаций

        # Параметры модели оптимизации
        self.optimization = optimization.lower()
        self.tau = tau  # параметр неуверенности в равновесных доходностях
        self.market_caps = market_caps  # <-- Добавлено присваивание параметра
        self.views = views  # субъективные прогнозы
        self.view_confidences = view_confidences  # уверенность в прогнозах
        
        # Дополнительные атрибуты для Блэка-Литермана
        self.market_weights = None  # веса рыночного портфеля
        self.implied_returns = None  # равновесные доходности
        self.posterior_returns = None  # апостериорные доходности

        # сука костыль
        self.include_short_selling = include_short_selling
        
        # Проверка выбранной модели оптимизации
        if self.optimization not in ['markowitz', 'black_litterman']:
            self.logger.warning(f"Неизвестная модель оптимизации: {optimization}. Будет использована модель Марковица.")
            self.optimization = 'markowitz'
            
        self.logger.info(f"Инициализирован оптимизатор портфеля с моделью {self.optimization.upper()}")


    def load_risk_free_portfolio(self):
        """
        Загружает данные о портфеле облигаций для безрисковой части
        """
        if not self.risk_free_portfolio_file:
            self.logger.info("Файл с портфелем облигаций не указан, используется обобщенная безрисковая ставка")
            return None
            
        if not os.path.exists(self.risk_free_portfolio_file):
            self.logger.warning(f"Файл с портфелем облигаций не найден: {self.risk_free_portfolio_file}")
            return None
            
        try:
            self.logger.info(f"Загрузка портфеля облигаций из {self.risk_free_portfolio_file}")
            bonds_df = pd.read_csv(self.risk_free_portfolio_file)
            
            # Проверяем наличие необходимых колонок
            required_columns = ['security_code', 'full_name', 'weight']
            if not all(col in bonds_df.columns for col in required_columns):
                self.logger.warning(f"В файле с облигациями отсутствуют необходимые колонки: {required_columns}")
                return None
                
            # Нормализуем веса, если они представлены в процентах (0-100)
            if bonds_df['weight'].max() > 1:
                bonds_df['weight'] = bonds_df['weight'] / 100
                
            # Проверяем, что сумма весов примерно равна 1
            total_weight = bonds_df['weight'].sum()
            if not 0.99 <= total_weight <= 1.01:
                self.logger.warning(f"Сумма весов облигаций ({total_weight}) не равна 1. Выполняем нормализацию.")
                bonds_df['weight'] = bonds_df['weight'] / total_weight
                
            self.risk_free_portfolio = bonds_df
            self.logger.info(f"Загружен портфель из {len(bonds_df)} облигаций")
            return bonds_df
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке портфеля облигаций: {e}")
            return None

        
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
        """Выбирает метод подготовки доходностей в зависимости от флага"""
        if self.include_short_selling:
            return self.prepare_returns_with_shorts()
        else:
            return self.prepare_returns_standard()
    
    def prepare_returns_standard(self):
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
        
    def prepare_returns_with_shorts(self):
        """Подготовка доходностей с учетом коротких позиций"""
        if self.df is None:
            self.logger.error("Данные не загружены")
            return None
            
        self.logger.info("Подготовка данных доходностей для оптимизации с учетом шортов")
        
        try:
            # Фильтрация только по shortlist, без фильтрации по неотрицательным сигналам
            filtered_df = self.df
            
            if 'in_shortlist' in self.df.columns:
                filtered_df = self.df[self.df['in_shortlist'] == True]
                self.logger.info(f"Отфильтровано {len(filtered_df)} строк по shortlist")

            # Если есть колонка ticker, группируем по тикерам
            if 'ticker' in filtered_df.columns:
                # Для каждого тикера анализируем сигналы
                returns_dict = {}
                
                for ticker, ticker_data in filtered_df.groupby('ticker'):
                    ticker_data = ticker_data.sort_index()
                    
                    # Проверяем, есть ли колонка signal
                    if 'signal' in ticker_data.columns:
                        # Подсчитываем частоту различных сигналов
                        signal_counts = ticker_data['signal'].value_counts()
                        
                        # Проверяем, что есть хотя бы один сигнал
                        if len(signal_counts) > 0:
                            # Берем сигнал с наибольшей частотой
                            predominant_signal = signal_counts.idxmax()
                            
                            # Проверяем, что есть колонка close
                            if 'close' in ticker_data.columns:
                                returns = ticker_data['close'].pct_change().dropna()
                                
                                # Для коротких позиций инвертируем доходность
                                if predominant_signal < 0:
                                    returns = -returns
                                    self.logger.info(f"Тикер {ticker} добавлен с КОРОТКОЙ позицией")
                                else:
                                    self.logger.info(f"Тикер {ticker} добавлен с ДЛИННОЙ позицией")
                                
                                returns_dict[ticker] = returns
                        else:
                            self.logger.info(f"Тикер {ticker} пропущен: нет преобладающего сигнала")
                    else:
                        self.logger.info(f"Тикер {ticker} пропущен: нет колонки signal")
                        
                # Объединение всех доходностей в одну таблицу
                if returns_dict:
                    self.returns = pd.DataFrame(returns_dict)
                    self.logger.info(f"Рассчитаны доходности для {len(returns_dict)} тикеров")
                else:
                    self.logger.warning("Нет подходящих тикеров для создания портфеля")
                    return None
            else:
                self.logger.error("Колонка 'ticker' не найдена в данных")
                return None
                
            return self.returns
                
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке доходностей с учетом шортов: {e}")
            return None
    
    def load_market_caps(self):
        """
        Загружает реальные данные о рыночной капитализации для тикеров
        из {BASE_PATH}/data/processed_data/{ticker}/market_cap/cap.csv
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны, невозможно загрузить рыночные капитализации")
            return None
        
        self.logger.info("Загрузка данных о рыночной капитализации")
        market_caps = {}
        
        for ticker in self.returns.columns:
            cap_file = f"{BASE_PATH}/data/processed_data/{ticker}/market_cap/cap.csv"
            
            try:
                if os.path.exists(cap_file):
                    # Загружаем файл с капитализацией
                    cap_data = pd.read_csv(cap_file)
                    
                    # Используем последнее доступное значение
                    if len(cap_data) > 0:
                        # Предполагаем, что у нас есть колонки 'date' и 'market_cap'
                        if 'date' in cap_data.columns and 'market_cap' in cap_data.columns:
                            # Сортируем по дате и берем последнюю
                            cap_data = cap_data.sort_values('date', ascending=False)
                            market_cap = cap_data.iloc[0]['market_cap']
                            
                            # Преобразуем текстовое представление в число
                            if isinstance(market_cap, str):
                                if "трлн" in market_cap:
                                    value = float(market_cap.replace("трлн", "").strip()) * 1000
                                elif "млрд" in market_cap:
                                    value = float(market_cap.replace("млрд", "").strip())
                                elif "млн" in market_cap:
                                    value = float(market_cap.replace("млн", "").strip()) / 1000
                                else:
                                    value = float(market_cap)
                            else:
                                value = float(market_cap)
                                
                            market_caps[ticker] = value
                            self.logger.info(f"Загружена капитализация для {ticker}: {value} млрд")
                        else:
                            self.logger.warning(f"Некорректный формат файла {cap_file}")
            except Exception as e:
                self.logger.warning(f"Ошибка при загрузке капитализации для {ticker}: {e}")
        
        # Если не удалось загрузить ни одной капитализации, создаем синтетические
        if not market_caps:
            self.logger.warning("Не удалось загрузить реальные данные о капитализации, создаем синтетические")
            # Создаем синтетические данные на основе композитного скора, если он есть
            if self.df is not None and 'ticker' in self.df.columns and 'composite_score' in self.df.columns:
                # Группируем по тикерам и берем среднее значение композитного скора
                ticker_scores = self.df.groupby('ticker')['composite_score'].mean()
                
                for ticker in self.returns.columns:
                    if ticker in ticker_scores.index:
                        # Базовая капитализация 1000 млрд + премия за композитный скор
                        base_cap = 1000
                        score = ticker_scores[ticker]
                        score_premium = (score + 1) * 500  # Премия от 0 до 1000 млрд
                        market_caps[ticker] = base_cap + score_premium
                    else:
                        # Для тикеров без скора используем среднее значение
                        market_caps[ticker] = 1000
            else:
                # Если нет композитного скора, используем равные капитализации
                for ticker in self.returns.columns:
                    market_caps[ticker] = 1000
        
        self.market_caps = market_caps
        return market_caps
    
    def prepare_views_from_signals(self):
        """
        Подготавливает прогнозы (views) и уверенность в них на основе сигналов
        """
        if self.df is None or 'ticker' not in self.df.columns:
            self.logger.error("Нет данных с тикерами для создания прогнозов")
            return None, None
        
        self.logger.info("Подготовка прогнозов на основе сигналов")
        
        # Прогнозы и уверенность в них
        views = {}
        view_confidences = {}
        
        # Фильтруем только последние данные для каждого тикера, если есть дата
        if 'date' in self.df.columns or isinstance(self.df.index, pd.DatetimeIndex):
            # Определяем максимальную дату
            if 'date' in self.df.columns:
                latest_date = self.df['date'].max()
                latest_data = self.df[self.df['date'] == latest_date]
            else:
                latest_date = self.df.index.max()
                latest_data = self.df[self.df.index == latest_date]
        else:
            latest_data = self.df
        
        # Фильтруем только те, что в шортлисте (если есть такая колонка)
        if 'in_shortlist' in latest_data.columns:
            filtered_data = latest_data[latest_data['in_shortlist'] == True]
        else:
            filtered_data = latest_data
        
        # Используем только тикеры, для которых у нас есть исторические доходности
        if self.returns is not None:
            available_tickers = set(self.returns.columns)
            filtered_data = filtered_data[filtered_data['ticker'].isin(available_tickers)]
        
        # Создаем прогнозы на основе композитного скора
        for _, row in filtered_data.iterrows():
            ticker = row['ticker']
            
            # Базовая ожидаемая доходность рынка (10%)
            market_return = 0.10
            
            # Используем композитный скор для создания прогноза
            composite_score = row.get('composite_score', 0)
            
            # Масштабируем скор [-1, 1] в премию [-10%, +20%]
            score_premium = composite_score * (0.20 if composite_score > 0 else 0.10)
            
            # Итоговый прогноз доходности (не меньше безрисковой ставки)
            expected_return = max(self.risk_free_rate, market_return + score_premium)
            
            views[ticker] = expected_return
            
            # Рассчитываем уверенность на основе силы сигнала и фундаментального скора
            signal_confidence = 0.5  # Базовая уверенность
            
            # Если есть сигнал покупки, увеличиваем уверенность
            if 'signal' in row and row['signal'] == 1:
                signal_confidence += 0.2
            
            # Если есть сильный фундаментальный скор, также увеличиваем уверенность
            if 'fundamental_score' in row:
                fund_score = row['fundamental_score']
                fund_confidence = 0.3 * abs(fund_score) if fund_score > 0 else 0
                signal_confidence += fund_confidence
            
            # Ограничиваем уверенность в диапазоне [0.3, 0.9]
            view_confidences[ticker] = min(0.9, max(0.3, signal_confidence))
        
        self.views = views
        self.view_confidences = view_confidences
        
        return views, view_confidences
    
    def calculate_market_weights(self):
        """
        Рассчитывает рыночные веса на основе капитализации для модели Блэка-Литермана
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
        
        # Если капитализации не загружены, загружаем их
        if self.market_caps is None:
            self.load_market_caps()
        
        # Если все еще нет капитализаций, используем равные веса
        if not self.market_caps:
            self.logger.warning("Не удалось загрузить рыночные капитализации, используются равные веса")
            n_assets = len(self.returns.columns)
            self.market_weights = pd.Series(
                [1/n_assets] * n_assets, 
                index=self.returns.columns
            )
            return self.market_weights
        
        try:
            # Создаем Series с капитализациями для тикеров из доходностей
            market_caps_series = pd.Series({
                ticker: self.market_caps.get(ticker, 0) 
                for ticker in self.returns.columns
            })
            
            # Проверяем, что у всех тикеров есть капитализация
            missing_tickers = market_caps_series[market_caps_series == 0].index
            if len(missing_tickers) > 0:
                self.logger.warning(f"Отсутствует капитализация для {len(missing_tickers)} тикеров: {missing_tickers}")
                
                # Для отсутствующих тикеров берем среднюю капитализацию
                mean_cap = market_caps_series[market_caps_series > 0].mean()
                for ticker in missing_tickers:
                    market_caps_series[ticker] = mean_cap
            
            # Рассчитываем веса на основе капитализации
            self.market_weights = market_caps_series / market_caps_series.sum()
            
            self.logger.info(f"Рассчитаны рыночные веса на основе капитализации для {len(self.market_weights)} активов")
            return self.market_weights
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете рыночных весов: {e}")
            # Используем равные веса как запасной вариант
            n_assets = len(self.returns.columns)
            self.market_weights = pd.Series([1/n_assets] * n_assets, index=self.returns.columns)
            return self.market_weights
    
    def calculate_implied_returns(self):
        """
        Рассчитывает равновесные доходности на основе CAPM для модели Блэка-Литермана
        """
        if self.returns is None:
            self.logger.error("Доходности не рассчитаны")
            return None
            
        if self.market_weights is None:
            self.calculate_market_weights()
            
        self.logger.info("Расчет равновесных доходностей на основе CAPM")
        
        try:
            # Годовая ковариационная матрица доходностей
            cov_matrix = self.returns.cov() * 252
            
            # Используем формулу равновесных доходностей: π = δΣw
            # где δ - коэффициент неприятия риска (можно использовать коэффициент Шарпа рынка)
            # Σ - ковариационная матрица, w - веса рыночного портфеля
            
            # Рассчитаем приближенное значение коэффициента неприятия риска
            market_volatility = np.sqrt(self.market_weights.dot(cov_matrix).dot(self.market_weights))
            risk_aversion = (self.risk_free_rate + 0.05) / market_volatility  # Используем рыночную премию 5%
            
            # Рассчитываем равновесные доходности
            self.implied_returns = risk_aversion * cov_matrix.dot(self.market_weights)
            
            self.logger.info(f"Рассчитаны равновесные доходности для {len(self.implied_returns)} активов")
            return self.implied_returns
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете равновесных доходностей: {e}")
            return None
            
    def incorporate_views(self):
        """
        Включает субъективные прогнозы в модель Блэка-Литермана
        """
        if self.implied_returns is None:
            self.logger.error("Равновесные доходности не рассчитаны")
            return None
        
        # Если прогнозы не предоставлены, пытаемся создать их из сигналов
        if self.views is None:
            self.prepare_views_from_signals()
            
        if self.views is None or len(self.views) == 0:
            self.logger.warning("Не удалось создать прогнозы, используются только равновесные доходности")
            self.posterior_returns = self.implied_returns.copy()
            return self.posterior_returns
            
        self.logger.info("Включение субъективных прогнозов в модель")
        
        try:
            # Преобразуем views в словарь, если это список
            views_dict = self.views
            if isinstance(self.views, list):
                views_dict = {ticker: return_value for ticker, return_value in self.views}
                
            # Преобразуем view_confidences в словарь, если это список
            confidences_dict = self.view_confidences or {}
            if isinstance(self.view_confidences, list):
                confidences_dict = {ticker: conf for ticker, conf in self.view_confidences}
                
            # Фильтруем только те активы, которые есть в наших доходностях
            views_filtered = {ticker: return_value for ticker, return_value in views_dict.items() 
                             if ticker in self.returns.columns}
            
            if len(views_filtered) < len(views_dict):
                missing_tickers = set(views_dict.keys()) - set(self.returns.columns)
                self.logger.warning(f"Игнорируются прогнозы для {len(missing_tickers)} тикеров: {missing_tickers}")
            
            # Если нет прогнозов для наших активов, используем равновесные доходности
            if not views_filtered:
                self.logger.warning("Не найдены применимые прогнозы, используются только равновесные доходности")
                self.posterior_returns = self.implied_returns.copy()
                return self.posterior_returns
            
            # Годовая ковариационная матрица доходностей
            cov_matrix = self.returns.cov() * 252
            
            # Создаем матрицу выборки P для представления прогнозов
            tickers = self.returns.columns
            P = np.zeros((len(views_filtered), len(tickers)))
            q = np.zeros(len(views_filtered))
            omega = np.zeros((len(views_filtered), len(views_filtered)))
            
            # Заполняем матрицы для каждого прогноза
            for i, (ticker, expected_return) in enumerate(views_filtered.items()):
                ticker_idx = list(tickers).index(ticker)
                P[i, ticker_idx] = 1  # Абсолютный прогноз для конкретного актива
                q[i] = expected_return
                
                # Уверенность в прогнозе (по умолчанию 0.5 - средняя уверенность)
                confidence = confidences_dict.get(ticker, 0.5)
                # Диагональные элементы omega - обратно пропорциональны уверенности
                omega[i, i] = (1 - confidence) * (P[i] @ cov_matrix @ P[i].T)
            
            # Формула Блэка-Литермана для апостериорных доходностей
            # E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)q]
            
            # Инвертируем матрицы, обрабатывая возможные ошибки
            try:
                tau_sigma_inv = np.linalg.inv(self.tau * cov_matrix.values)
                omega_inv = np.linalg.inv(omega)
            except np.linalg.LinAlgError:
                self.logger.warning("Не удалось инвертировать матрицы, используем псевдообратную матрицу")
                tau_sigma_inv = np.linalg.pinv(self.tau * cov_matrix.values)
                omega_inv = np.linalg.pinv(omega)
            
            # Рассчитываем апостериорные доходности
            term1 = tau_sigma_inv + P.T @ omega_inv @ P
            term2 = tau_sigma_inv @ self.implied_returns.values + P.T @ omega_inv @ q
            
            try:
                posterior_returns_values = np.linalg.inv(term1) @ term2
            except np.linalg.LinAlgError:
                self.logger.warning("Не удалось инвертировать term1, используем псевдообратную матрицу")
                posterior_returns_values = np.linalg.pinv(term1) @ term2
            
            # Создаем Series с апостериорными доходностями
            self.posterior_returns = pd.Series(posterior_returns_values, index=tickers)
            
            self.logger.info(f"Рассчитаны апостериорные доходности с учетом {len(views_filtered)} прогнозов")
            return self.posterior_returns
            
        except Exception as e:
            self.logger.error(f"Ошибка при включении прогнозов: {e}")
            # В случае ошибки используем равновесные доходности
            self.posterior_returns = self.implied_returns.copy()
            return self.posterior_returns
    
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

        n = len(returns.columns)
        
        if bounds is None and constrained:
            bounds = tuple((0, self.max_weight) for _ in range(n))
            
        self.logger.info(f"Запуск оптимизации портфеля для {returns.shape[1]} активов с моделью {self.optimization.upper()}")
        
        # Ограничения: сумма весов = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Границы весов: от 0 до max_weight для каждого актива
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(n))
        
        # Начальные веса: равные для всех активов
        init_weights = np.array([1/n] * n)
        
        try:
            # Для модели Блэка-Литермана используем рассчитанные апостериорные доходности
            if self.optimization == 'black_litterman':
                self.logger.info("Применение модели Блэка-Литермана")
                
                # Расчет равновесных доходностей
                if self.implied_returns is None:
                    self.calculate_implied_returns()
                
                # Включение прогнозов
                if self.posterior_returns is None:
                    self.incorporate_views()
                
                # Для оптимизации используем апостериорные ожидаемые доходности
                # вместо средних исторических доходностей
                
                # Создаем копию исторических доходностей
                bl_returns = returns.copy()
                
                # Заменяем средние исторические доходности на апостериорные
                expected_returns_annualized = self.posterior_returns / 252
                
                # Оптимизация с использованием апостериорных доходностей
                # Создаем функцию для минимизации с учетом апостериорных доходностей
                def bl_objective(weights):
                    # Годовая доходность на основе апостериорных ожиданий
                    portfolio_return = np.sum(self.posterior_returns * weights)
                    # Годовая волатильность на основе исторической ковариации
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
                    # Отрицательный коэффициент Шарпа
                    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                    return -sharpe
                
                # Оптимизация
                results = minimize(bl_objective, 
                                   init_weights, 
                                   method='SLSQP', 
                                   bounds=bounds if constrained else None,
                                   constraints=constraints)
            else:
                # Стандартная оптимизация Марковица
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
        
        # Загружаем детализированный портфель облигаций, если еще не загружен
        if self.risk_free_portfolio is None and self.risk_free_portfolio_file:
            self.load_risk_free_portfolio()
        
        # Детальная информация о безрисковой части
        rf_details = {}
        
        # Добавление безрисковой части в веса
        if self.risk_free_portfolio is not None:
            # Если есть детализированный портфель облигаций, добавляем каждую облигацию
            for _, bond in self.risk_free_portfolio.iterrows():
                bond_name = bond['full_name']
                bond_weight = bond['weight'] * rf_allocation
                # Добавляем информацию об облигациях, но не в основную структуру весов
                rf_details[bond_name] = {
                    'security_code': bond['security_code'],
                    'weight': bond_weight,
                    'original_weight': bond['weight'],
                    'yield': bond.get('yield', self.risk_free_rate * 100)
                }
            
            # В основные веса добавляем только общую безрисковую часть
            final_weights['RISK_FREE'] = rf_allocation
        else:
            # Иначе добавляем общую безрисковую часть
            final_weights['RISK_FREE'] = rf_allocation
        
        # Расчет ожидаемой доходности и волатильности
        if self.optimization == 'black_litterman' and self.posterior_returns is not None:
            # Для модели Блэка-Литермана используем апостериорные доходности
            expected_return = np.sum(self.posterior_returns * self.optimal_weights)
            expected_volatility = np.sqrt(np.dot(self.optimal_weights.T, 
                                    np.dot(self.returns.cov() * 252, self.optimal_weights)))
        else:
            # Для модели Марковица - стандартный расчет
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
            'rf_allocation': rf_allocation,
            'optimization_model': self.optimization,
            'rf_details': rf_details  # Добавляем детали по облигациям
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
        ax1.set_title(f'Эффективная граница и оптимальные портфели ({self.optimization.upper()})', fontsize=16)
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
        
        # Добавляем модель в имя файла
        model_suffix = self.optimization.lower()
        
        # Закрываем все предыдущие фигуры перед созданием новых
        plt.close('all')
        
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
        plt.title(f'Оптимальные веса портфеля ({self.optimization.upper()})')
        
        if output_dir:
            # Добавляем модель в имя файла!
            filename = f'portfolio_weights_pie_{model_suffix}.png'
            plt.savefig(os.path.join(output_dir, filename))
            self.logger.info(f"График весов сохранен в {output_dir}/{filename}")
        else:
            plt.show()
                
        plt.close()
        
        # Если есть детали по облигациям, создаем отдельный график для них
        rf_details = self.portfolio_performance.get('rf_details', {})
        if rf_details:
            plt.figure(figsize=(12, 8))
            
            # Подготавливаем данные для графика облигаций
            bond_names = []
            bond_weights = []
            
            for name, details in rf_details.items():
                bond_names.append(name)
                bond_weights.append(details['weight'])
            
            # Создаем круговую диаграмму для облигаций
            plt.pie(
                bond_weights, 
                labels=bond_names,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                explode=[0.05] * len(bond_names)
            )
            plt.axis('equal')
            plt.title(f'Структура безрисковой части портфеля ({self.optimization.upper()})')
            
            if output_dir:
                # Также добавляем модель в имя файла!
                bond_filename = f'bond_weights_pie_{model_suffix}.png'
                plt.savefig(os.path.join(output_dir, bond_filename))
                self.logger.info(f"График весов облигаций сохранен в {output_dir}/{bond_filename}")
            else:
                plt.show()
            
            plt.close()
 
        # Создание улучшенного графика эффективной границы (без изменений)
        try:
            self.plot_enhanced_efficient_frontier(output_dir=output_dir)
            self.logger.info("Улучшенный график эффективной границы успешно создан")
        except Exception as e:
            self.logger.error(f"Ошибка при создании улучшенного графика эффективной границы: {e}")
            try:
                self.plot_efficient_frontier(output_dir=output_dir)
                self.logger.info("Стандартный график эффективной границы успешно создан")
            except Exception as e2:
                self.logger.error(f"Ошибка при создании стандартного графика эффективной границы: {e2}")
        
        if output_dir:
            # Сохраняем основные веса портфеля
            weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
            weights_df.to_csv(os.path.join(output_dir, 'portfolio_weights.csv'), index=False)
            
            # Если есть детали по облигациям, сохраняем их в отдельный файл
            if rf_details:
                # Создаем DataFrame с деталями облигаций
                bond_data = []
                for name, details in rf_details.items():
                    bond_data.append({
                        'Bond': name,
                        'Security_Code': details['security_code'],
                        'Weight_In_Portfolio': details['weight'],
                        'Weight_In_Bond_Part': details['original_weight'],
                        'Yield': details.get('yield', self.risk_free_rate * 100)
                    })
                
                bond_df = pd.DataFrame(bond_data)
                bond_df.to_csv(os.path.join(output_dir, 'bond_details.csv'), index=False)
                self.logger.info(f"Детали облигаций сохранены в {output_dir}/bond_details.csv")
            
            # Сохраняем текстовый отчет
            with open(os.path.join(output_dir, 'portfolio_summary.txt'), 'w') as f:
                f.write(f"Ожидаемая годовая доходность: {self.portfolio_performance['expected_return']*100:.2f}%\n")
                f.write(f"Ожидаемая годовая волатильность: {self.portfolio_performance['expected_volatility']*100:.2f}%\n")
                f.write(f"Коэффициент Шарпа: {self.portfolio_performance['sharpe_ratio']:.2f}\n")
                f.write(f"Безрисковая ставка: {self.portfolio_performance['risk_free_rate']*100:.2f}%\n")
                f.write(f"Доля безрисковых активов: {self.portfolio_performance['rf_allocation']*100:.2f}%\n")
                f.write(f"Модель оптимизации: {self.optimization.upper()}\n\n")
                
                # Добавляем детали по облигациям, если они есть
                if rf_details:
                    f.write("ДЕТАЛИ БЕЗРИСКОВОЙ ЧАСТИ ПОРТФЕЛЯ:\n")
                    f.write("---------------------------------------\n")
                    for name, details in rf_details.items():
                        f.write(f"{name} ({details['security_code']}):\n")
                        f.write(f"  - Доля в портфеле: {details['weight']*100:.2f}%\n")
                        f.write(f"  - Доля в безрисковой части: {details['original_weight']*100:.2f}%\n")
                        f.write(f"  - Доходность: {details.get('yield', self.risk_free_rate*100):.2f}%\n")
            
            self.logger.info(f"Результаты сохранены в {output_dir}")
            
            # Сохраняем дополнительную информацию для модели Блэка-Литермана (без изменений)
            if self.optimization == 'black_litterman':
                bl_info = {
                    'market_weights': self.market_weights.to_dict() if self.market_weights is not None else None,
                    'implied_returns': self.implied_returns.to_dict() if self.implied_returns is not None else None,
                    'posterior_returns': self.posterior_returns.to_dict() if self.posterior_returns is not None else None,
                    'views': self.views,
                    'view_confidences': self.view_confidences,
                    'tau': self.tau
                }
                
                # Преобразуем все значения в нормальный json-совместимый формат
                bl_info_json = {}
                for key, value in bl_info.items():
                    if isinstance(value, dict):
                        bl_info_json[key] = {str(k): float(v) for k, v in value.items()}
                    elif value is None:
                        bl_info_json[key] = None
                    else:
                        bl_info_json[key] = value
                
                with open(os.path.join(output_dir, 'bl_parameters.json'), 'w') as f:
                    json.dump(bl_info_json, f, indent=4)
                
                self.logger.info(f"Параметры модели Блэка-Литермана сохранены в {output_dir}/bl_parameters.json")

    
    def run_pipeline(self, input_file=None, output_dir=f"{BASE_PATH}/data",
              risk_free_rate=None, min_rf_allocation=None, max_rf_allocation=None, max_weight=None,
              views=None, view_confidences=None, optimization=None):
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
        views : dict or list, optional
            Субъективные прогнозы для модели Блэка-Литермана
        view_confidences : dict or list, optional
            Уверенность в прогнозах для модели Блэка-Литермана
        optimization : str, optional
            Модель оптимизации: 'markowitz' или 'black_litterman'
        
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
        if views is not None:
            self.views = views
        if view_confidences is not None:
            self.view_confidences = view_confidences
        if optimization is not None:
            self.optimization = optimization.lower()
            if self.optimization not in ['markowitz', 'black_litterman']:
                self.logger.warning(f"Неизвестная модель оптимизации: {optimization}. Будет использована модель Марковица.")
                self.optimization = 'markowitz'
            
        self.logger.info(f"Максимальный вес одной бумаги: {self.max_weight}")
        self.logger.info(f"Используется модель оптимизации: {self.optimization.upper()}")
            
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
            
        # Для модели Блэка-Литермана выполняем предварительные расчеты
        if self.optimization == 'black_litterman':
            self.load_market_caps()
            self.calculate_market_weights()
            self.calculate_implied_returns()
            self.prepare_views_from_signals()
            self.incorporate_views()
            
        # Оптимизация портфеля с учетом max_weight
        self.optimize_portfolio(constrained=True, 
                            bounds=tuple((0, self.max_weight) for _ in range(len(self.returns.columns))))
        
        if self.optimal_weights is None:
            self.logger.error("Не удалось оптимизировать портфель")
            return None
            
        # Расчет итогового портфеля с безрисковой частью
        portfolio = self.calculate_final_portfolio()
        
        # Создаем подпапку в зависимости от модели оптимизации
        model_subdir = 'markowitz' if self.optimization == 'markowitz' else 'black_litterman'
        model_output_dir = os.path.join(output_dir, 'portfolio', model_subdir)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Визуализация результатов
        try:
            self.visualize_portfolio(model_output_dir)
        except Exception as e:
            self.logger.error(f"Ошибка при визуализации: {e}")
        finally:
            # Дополнительно закрываем все фигуры после визуализации
            plt.close('all')
                
        return portfolio

def run_all_optimization_models(
    base_path,
    tickers_list,
    risk_free_rate=0.1,
    min_rf_allocation=0.3,
    max_rf_allocation=0.5,
    max_weight=0.15,
    input_file=None,
    risk_free_portfolio_file="/Users/aeshef/Documents/GitHub/kursach/data/processed_data/BONDS/kbd/portfolios/bond_portfolio_20250426.csv",
    include_short_selling=False
):
    """
    Запускает последовательно все модели оптимизации портфеля
    
    Parameters:
    -----------
    base_path : str
        Базовый путь проекта
    tickers_list : list
        Список тикеров для анализа
    risk_free_rate : float
        Безрисковая ставка
    min_rf_allocation : float
        Минимальная доля безрисковых активов
    max_rf_allocation : float
        Максимальная доля безрисковых активов
    max_weight : float
        Максимальный вес одной бумаги в портфеле
    input_file : str, optional
        Путь к файлу с сигналами (если None, используется стандартный путь)
    risk_free_portfolio_file : str, optional
        Путь к файлу с оптимальным портфелем облигаций
        
    Returns:
    --------
    dict
        Словарь с результатами оптимизации для обеих моделей
    """
    from pys.utils.logger import BaseLogger
    from pys.data_collection.market_cap import run_pipeline_market_cap
    logger = BaseLogger('OptimizationPipeline').logger
    import pandas as pd 
    
    # Задаем пути к файлам
    if input_file is None:
        input_file = f"{base_path}/data/signals.csv"
        
    output_dir = f"{base_path}/data"
    
    # Запустим парсер капитализаций для модели Блэка-Литермана
    logger.info("Запуск парсера капитализаций для модели Блэка-Литермана")

    market_caps_df = run_pipeline_market_cap(
        base_path=base_path,
        tickers=tickers_list
    )
    
    # Подготовим данные для модели Блэка-Литермана
    market_caps = dict(zip(market_caps_df['ticker'], market_caps_df['market_cap']))
    
    # Результаты будем хранить в словаре
    results = {}
    
    # 1. Запуск модели Марковица
    logger.info("Запуск оптимизации по модели Марковица")
    optimizer_markowitz = PortfolioOptimizer(
        risk_free_rate=risk_free_rate,
        min_rf_allocation=min_rf_allocation,
        max_rf_allocation=max_rf_allocation,
        max_weight=max_weight,
        optimization='markowitz',
        risk_free_portfolio_file=risk_free_portfolio_file,
        include_short_selling=include_short_selling
    )
    
    results['markowitz'] = optimizer_markowitz.run_pipeline(
        input_file=input_file,
        output_dir=output_dir
    )
    
    # 2. Запуск модели Блэка-Литермана
    logger.info("Запуск оптимизации по модели Блэка-Литермана")
    optimizer_bl = PortfolioOptimizer(
        risk_free_rate=risk_free_rate,
        min_rf_allocation=min_rf_allocation,
        max_rf_allocation=max_rf_allocation,
        max_weight=max_weight,
        optimization='black_litterman',
        market_caps=market_caps,
        risk_free_portfolio_file=risk_free_portfolio_file,
        include_short_selling=include_short_selling
    )
    
    results['black_litterman'] = optimizer_bl.run_pipeline(
        input_file=input_file,
        output_dir=output_dir
    )
    
    # Подготовка сравнительной таблицы результатов (без изменений)
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Создаем сравнительную таблицу
        comparison = pd.DataFrame({
            'Марковиц': results['markowitz']['weights'],
            'Блэк-Литерман': results['black_litterman']['weights']
        })
        
        # Удаляем строки с нулевыми весами в обеих моделях
        comparison = comparison[(comparison['Марковиц'] > 0) | (comparison['Блэк-Литерман'] > 0)]
        comparison = comparison.sort_values(by='Марковиц', ascending=False)
        
        # Сохраняем таблицу
        comparison_path = f"{output_dir}/portfolio/comparison.csv"
        comparison.to_csv(comparison_path)
        logger.info(f"Сравнительная таблица сохранена в {comparison_path}")
        
        # Создаем сравнительный график
        plt.figure(figsize=(12, 8))
        comparison.plot(kind='bar', figsize=(12, 8))
        plt.title('Сравнение весов портфелей: Марковиц vs Блэк-Литерман')
        plt.xlabel('Тикер')
        plt.ylabel('Вес в портфеле')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Сохраняем график
        comparison_plot_path = f"{output_dir}/portfolio/comparison_plot.png"
        plt.savefig(comparison_plot_path)
        plt.close()
        logger.info(f"Сравнительный график сохранен в {comparison_plot_path}")
        
        # Добавляем подробности о портфеле облигаций в сравнительный отчет
        if os.path.exists(risk_free_portfolio_file):
            try:
                bonds_df = pd.read_csv(risk_free_portfolio_file)
                bonds_summary_path = f"{output_dir}/portfolio/bonds_summary.csv"
                bonds_df.to_csv(bonds_summary_path, index=False)
                logger.info(f"Информация о портфеле облигаций сохранена в {bonds_summary_path}")
                
                # Создаем текстовый отчет с деталями о портфеле облигаций
                with open(f"{output_dir}/portfolio/bonds_details.txt", 'w') as f:
                    f.write("ДЕТАЛИ ПОРТФЕЛЯ ОБЛИГАЦИЙ\n")
                    f.write("==========================\n\n")
                    f.write(f"Доля безрисковых активов в портфеле: {min_rf_allocation*100:.1f}% - {max_rf_allocation*100:.1f}%\n")
                    f.write(f"Используемая безрисковая ставка: {risk_free_rate*100:.2f}%\n\n")
                    
                    f.write("Структура портфеля облигаций:\n")
                    for _, row in bonds_df.iterrows():
                        f.write(f"- {row['full_name']} ({row['security_code']})\n")
                        f.write(f"  Доходность: {row.get('yield', 'Н/Д')}%\n")
                        f.write(f"  Дюрация: {row.get('duration_years', 'Н/Д')} лет\n")
                        f.write(f"  Доля в безрисковой части: {row['weight']}%\n\n")
            except Exception as e:
                logger.error(f"Ошибка при создании отчета о портфеле облигаций: {e}")
        
    except Exception as e:
        logger.error(f"Ошибка при создании сравнительных материалов: {e}")
    
    return results
