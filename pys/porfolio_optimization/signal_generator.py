import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shutil
import sys

# sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection')
# from private_info import BASE_PATH

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class SignalGenerator(BaseLogger):
    def __init__(self, input_file=None, weight_tech=0.4, weight_sentiment=0.3, weight_fundamental=0.3, 
                threshold_buy=0.5, threshold_sell=-0.5, log_level=logging.INFO):
        """
        Генератор торговых сигналов на основе композитного скора.
        
        Parameters:
        -----------
        input_file : str, optional
            Путь к CSV-файлу с объединенными данными
        weight_tech : float
            Вес технических индикаторов в композитном скоре
        weight_sentiment : float
            Вес сентимент-индикаторов в композитном скоре
        weight_fundamental : float
            Вес фундаментальных индикаторов в композитном скоре
        """
        super().__init__('SignalGenerator')
        self.input_file = input_file
        self.weight_tech = weight_tech
        self.weight_sentiment = weight_sentiment
        self.weight_fundamental = weight_fundamental
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.df = None
        
        # # Настройка логгера
        # self.logger = logging.getLogger('signal_generator')
        # self.logger.setLevel(log_level)
        
        # # Создаем обработчик для записи в файл
        # log_dir = 'logs'
        # os.makedirs(log_dir, exist_ok=True)
        # file_handler = logging.FileHandler(f'{log_dir}/signal_generator_{datetime.now().strftime("%Y%m%d")}.log')
        
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
    
    def load_data(self, input_file=None):
        """Загрузка данных для анализа"""
        if input_file:
            self.input_file = input_file
            
        self.logger.info(f"Загрузка данных из {self.input_file}")
        
        try:
            self.df = pd.read_csv(self.input_file)
            
            # Преобразование даты в индекс
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df.set_index('date', inplace=True)
                
            self.logger.info(f"Загружено {len(self.df)} строк данных")
            return self.df
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            return None
    
    def calculate_composite_score(self):
        """Рассчитывает композитный скор с учетом фундаментальных данных"""
        if self.df is None:
            self.logger.error("Данные не загружены")
            return None
            
        # Технические индикаторы для скоринга
        tech_indicators = ['RSI_14', 'MACD_diff', 'Stoch_%K', 'CCI_20', 'Williams_%R_14', 'ROC_10']
        
        # Сентимент-индикаторы
        sentiment_indicators = ['sentiment_compound_median', 'sentiment_direction', 
                               'sentiment_ma_7d', 'sentiment_ratio', 'sentiment_zscore_7d']
        
        self.logger.info("Расчет композитного скора")
        
        # Проверка наличия необходимых колонок
        available_tech = [col for col in tech_indicators if col in self.df.columns]
        available_sentiment = [col for col in sentiment_indicators if col in self.df.columns]
        
        # Расчет технического скора (без изменений)
        if not available_tech:
            self.logger.warning("Не найдены технические индикаторы")
            self.df['tech_score'] = 0
        else:
            # Нормализация технических индикаторов
            df_tech = self.df[available_tech].copy()
            # Инвертируем индикаторы, где меньшее значение лучше
            if 'Williams_%R_14' in df_tech.columns:
                df_tech['Williams_%R_14'] = -df_tech['Williams_%R_14']  
            
            # Стандартизация всех индикаторов
            scaler = StandardScaler()
            df_tech_scaled = pd.DataFrame(
                scaler.fit_transform(df_tech.fillna(df_tech.mean())),
                columns=df_tech.columns,
                index=df_tech.index
            )
            
            # Технический скор (среднее по всем техническим индикаторам)
            self.df['tech_score'] = df_tech_scaled.mean(axis=1)
            self.logger.info(f"Технический скор рассчитан на основе {len(available_tech)} индикаторов")
        
        # Расчет сентимент-скора (без изменений)
        if not available_sentiment:
            self.logger.warning("Не найдены сентимент-индикаторы")
            self.df['sentiment_score'] = 0
        else:
            # Сентимент-скор
            df_sentiment = self.df[available_sentiment].copy()
            # Стандартизация сентимент-индикаторов
            scaler = StandardScaler()
            df_sentiment_scaled = pd.DataFrame(
                scaler.fit_transform(df_sentiment.fillna(df_sentiment.mean())),
                columns=df_sentiment.columns,
                index=df_sentiment.index
            )
            self.df['sentiment_score'] = df_sentiment_scaled.mean(axis=1)
            self.logger.info(f"Сентимент-скор рассчитан на основе {len(available_sentiment)} индикаторов")
        
        # НОВЫЙ КОД: Расчет фундаментального скора
        self.logger.info("Расчет фундаментального скора")
        
        # Загружаем фундаментальные данные и рассчитываем квантили
        fund_data = self.load_fundamental_data()
        quantiles = self.calculate_quantiles(fund_data)
        
        # Создаем отчет о доступности данных
        self.logger.info("Доступность фундаментальных показателей:")
        for year in ['2023', '2024']:
            for indicator, stats in quantiles[year].items():
                data_ratio = stats.get('data_ratio', 0) * 100
                self.logger.info(f"  {year} - {indicator}: {data_ratio:.1f}% компаний")
        
        # Инициализируем фундаментальный скор
        self.df['fundamental_score'] = 0.0
        
        # Для каждого тикера и даты определяем фундаментальный скор
        if 'ticker' in self.df.columns:
            # Создаем словарь фундаментальных скоров для кэширования
            ticker_scores = {'2023': {}, '2024': {}}
            
            for year in ['2023', '2024']:
                for ticker in fund_data[year]:
                    ticker_scores[year][ticker] = self.calculate_ticker_fundamental_score(
                        ticker, year, fund_data, quantiles
                    )
            
            # Применяем скоры к DataFrame
            ticker_year_scores = {}  # для отчетности
            
            for index, row in self.df.iterrows():
                ticker = row['ticker']
                date = index if isinstance(index, pd.Timestamp) else pd.to_datetime(index)
                
                # Используем данные предыдущего года
                if date.year == 2024 and ticker in ticker_scores['2023']:
                    score = ticker_scores['2023'][ticker]
                    self.df.loc[index, 'fundamental_score'] = score
                    ticker_year_scores.setdefault(ticker, {})['2023'] = score
                
                elif date.year == 2025 and ticker in ticker_scores['2024']:
                    score = ticker_scores['2024'][ticker]
                    self.df.loc[index, 'fundamental_score'] = score
                    ticker_year_scores.setdefault(ticker, {})['2024'] = score
            
            # Отчет по фундаментальным скорам
            self.logger.info("Фундаментальные скоры по тикерам:")
            for ticker, years in ticker_year_scores.items():
                scores_str = ", ".join([f"{y}: {s:.3f}" for y, s in years.items()])
                self.logger.info(f"  {ticker}: {scores_str}")
        
        # Композитный скор на основе весов
        self.df['composite_score'] = (
            self.df['tech_score'] * self.weight_tech + 
            self.df['sentiment_score'] * self.weight_sentiment + 
            self.df['fundamental_score'] * self.weight_fundamental
        )
        
        self.logger.info("Композитный скор рассчитан")
        return self.df
    
    def load_fundamental_data(self, base_path=f'{BASE_PATH}/data/processed_data'):
        """Загружает фундаментальные данные для всех тикеров с обработкой NO_DATA"""
        if 'ticker' not in self.df.columns:
            self.logger.error("Колонка 'ticker' не найдена в данных")
            return {}
            
        # Получаем список уникальных тикеров
        tickers = self.df['ticker'].unique()
        self.logger.info(f"Загрузка фундаментальных данных для {len(tickers)} тикеров")
        
        # Хранилище для данных по годам
        fund_data = {'2023': {}, '2024': {}}
        
        # Загружаем данные для всех тикеров
        for ticker in tickers:
            for year in ['2023', '2024']:
                # Обновленный путь к файлу
                file_path = os.path.join(base_path, ticker, 'fundamental_analysis', year, 'common.csv')
                if os.path.exists(file_path):
                    try:
                        data = pd.read_csv(file_path)
                        
                        # Преобразуем значения NO_DATA в NaN
                        data['value_float'] = data['value_float'].apply(
                            lambda x: np.nan if x == 'NO_DATA' else float(x) 
                            if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())
                            else np.nan
                        )
                        
                        fund_data[year][ticker] = data
                        valid_count = data['value_float'].notna().sum()
                        missing_count = data['value_float'].isna().sum()
                        self.logger.info(f"Загружены данные {ticker} за {year}: {valid_count} показателей, {missing_count} отсутствует")
                    except Exception as e:
                        self.logger.warning(f"Ошибка загрузки {ticker} за {year}: {e}")
                else:
                    self.logger.warning(f"Файл не найден: {file_path}")
        
        return fund_data
    
    def calculate_quantiles(self, fund_data):
        """Рассчитывает квантили для каждого показателя, учитывая только реальные данные"""
        quantiles = {'2023': {}, '2024': {}}
        indicators_count = {'2023': {}, '2024': {}}
        
        for year in ['2023', '2024']:
            # Собираем все значения каждого показателя по всем компаниям
            all_indicators = {}
            company_count = len(fund_data[year])
            
            # Подсчет тикеров с данными по каждому показателю
            for ticker, data in fund_data[year].items():
                for _, row in data.iterrows():
                    indicator = row['Показатель']
                    value = row['value_float']
                    
                    if indicator not in indicators_count[year]:
                        indicators_count[year][indicator] = 0
                    
                    if not pd.isna(value):
                        indicators_count[year][indicator] += 1
                        
                        if indicator not in all_indicators:
                            all_indicators[indicator] = []
                        all_indicators[indicator].append(value)
            
            # Рассчитываем статистики для каждого показателя
            for indicator, values in all_indicators.items():
                if len(values) >= 3:  # Нужно минимум 3 значения
                    values = pd.Series(values)
                    quantiles[year][indicator] = {
                        'min': values.min(),
                        'max': values.max(),
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'data_ratio': indicators_count[year][indicator] / company_count  # процент компаний с данными
                    }
            
        return quantiles
    
    def calculate_ticker_fundamental_score(self, ticker, year, fund_data, quantiles):
        """Рассчитывает общий фундаментальный скор для тикера за определенный год"""
        if ticker not in fund_data[year]:
            return 0
            
        data = fund_data[year][ticker]
        
        # Веса показателей
        weights = {
            "Чистая прибыль, млрд руб": 0.10,
            "Див доход, ао, %": 0.10,
            "Дивиденды/прибыль, %": 0.05,
            "EBITDA, млрд руб": 0.08,
            "FCF, млрд руб": 0.10,
            "Рентаб EBITDA, %": 0.08,
            "Чистый долг, млрд руб": 0.08,
            "Долг/EBITDA": 0.07,
            "EPS, руб": 0.07,
            "ROE, %": 0.10,
            "ROA, %": 0.08,
            "P/E": 0.09
        }
        
        indicator_scores = []
        indicator_weights = []
        
        # Счетчики для отчетности
        total_indicators = 0
        valid_indicators = 0
        
        for _, row in data.iterrows():
            indicator = row['Показатель']
            value = row['value_float']
            
            if indicator in weights:
                total_indicators += 1
                
                weight = weights[indicator]
                
                if pd.isna(value):
                    # Для NO_DATA используем нейтральную оценку (0)
                    indicator_scores.append(0)
                    indicator_weights.append(weight * 0.5)  # уменьшаем вес для отсутствующих данных
                else:
                    score = self.calculate_indicator_score(indicator, value, quantiles, year)
                    indicator_scores.append(score * weight)
                    indicator_weights.append(weight)
                    valid_indicators += 1
        
        if not indicator_weights:
            return 0
            
        # Рассчитываем взвешенный скор
        weighted_score = sum(indicator_scores) / sum(indicator_weights)
        
        # Учитываем количество доступных показателей в общей оценке
        coverage_ratio = valid_indicators / max(1, total_indicators)
        confidence_factor = min(1.0, coverage_ratio * 1.5)  # Максимум 1.0, при покрытии 67%+
        
        return weighted_score * confidence_factor
    
    def calculate_indicator_score(self, indicator, value, quantiles, year):
        """Рассчитывает скор для отдельного фундаментального показателя с учетом NO_DATA"""
        if indicator not in quantiles[year]:
            return 0
            
        # Если значение отсутствует, устанавливаем его на медиану
        if pd.isna(value):
            return 0  # нейтральная оценка для отсутствующих данных
            
        # Определяем, хорошо ли когда показатель больше или меньше
        positive_indicators = [
            "Чистая прибыль, млрд руб", "Див доход, ао, %", "Дивиденды/прибыль, %", 
            "EPS, руб", "ROE, %", "ROA, %", "FCF, млрд руб", "Доходность FCF, %",
            "Операционная прибыль, млрд руб", "EBITDA, млрд руб", "Рентаб EBITDA, %",
            "Чистая рентаб, %", "Опер.денежный поток, млрд руб"
        ]
        
        negative_indicators = [
            "Опер. расходы, млрд руб", "Чистый долг, млрд руб", "P/E", "Долг/EBITDA",
            "CAPEX/Выручка, %"
        ]
        
        # Определяем направление - больше = лучше (True) или меньше = лучше (False)
        is_positive = indicator in positive_indicators or not indicator in negative_indicators
        
        q_data = quantiles[year][indicator]
        
        # Z-скор для нормализации
        if q_data['std'] > 0:
            z_score = (value - q_data['mean']) / q_data['std']
            # Ограничиваем z-скор в пределах [-3, 3]
            z_score = max(min(z_score, 3), -3)
        else:
            # Если нет разброса, используем относительную позицию
            if q_data['max'] > q_data['min']:
                relative_pos = (value - q_data['min']) / (q_data['max'] - q_data['min'])
                z_score = (relative_pos - 0.5) * 6  # масштабируем к [-3, 3]
            else:
                z_score = 0
                
        # Нормализуем до [-1, 1]
        score = z_score / 3
        
        # Инвертируем, если индикатор отрицательный
        if not is_positive:
            score = -score
            
        return score
    
    def generate_signals(self):
        """
        Генерирует торговые сигналы на основе композитного скора
        """
        if self.df is None or 'composite_score' not in self.df.columns:
            self.logger.error("Композитный скор не рассчитан")
            return None
            
        # Создаем новую колонку для сигналов (1 = Buy, 0 = Hold, -1 = Sell)
        self.df['signal'] = 0  # По умолчанию Hold
        
        # Buy сигнал - если композитный скор выше порога
        self.df.loc[self.df['composite_score'] > self.threshold_buy, 'signal'] = 1
        
        # Sell сигнал - если композитный скор ниже порога
        self.df.loc[self.df['composite_score'] < self.threshold_sell, 'signal'] = -1
        
        buy_count = (self.df['signal'] == 1).sum()
        hold_count = (self.df['signal'] == 0).sum()
        sell_count = (self.df['signal'] == -1).sum()
        
        self.logger.info(f"Сгенерировано сигналов: Buy - {buy_count}, Hold - {hold_count}, Sell - {sell_count}")
        return self.df
    
    def create_shortlist(self, top_pct=0.3):
        """
        Создает shortlist из топовых акций по композитному скору
        """
        if self.df is None or 'composite_score' not in self.df.columns:
            self.logger.error("Композитный скор не рассчитан")
            return None
            
        # Формирование shortlist из топ-акций (в каждый день)
        if 'ticker' in self.df.columns:
            # Колонка для отметки акций в shortlist
            self.df['in_shortlist'] = False
            
            for date, group in self.df.groupby(self.df.index.date):
                # Определение порога для топ X%
                threshold = group['composite_score'].quantile(1 - top_pct)
                self.df.loc[group[group['composite_score'] >= threshold].index, 'in_shortlist'] = True
            
            shortlist_count = self.df['in_shortlist'].sum()
            self.logger.info(f"Создан shortlist из {shortlist_count} записей (топ {top_pct*100}%)")
        else:
            self.logger.warning("Колонка 'ticker' не найдена, shortlist не создан")
            
        return self.df
    
    def save_shortlist(self, output_dir):
        """
        Сохраняет шортлист акций в CSV-файл
        """
        if self.df is None or 'in_shortlist' not in self.df.columns:
            self.logger.error("Шортлист не создан")
            return None
            
        # Создаем каталог для результатов, если его нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Получаем последнюю дату из данных
        latest_date = self.df.index.max().strftime('%Y%m%d')
        
        # Фильтруем данные только по последней дате и только те, что в шортлисте
        latest_df = self.df[self.df.index == self.df.index.max()]
        shortlist_df = latest_df[latest_df['in_shortlist'] == True].copy()
        
        if len(shortlist_df) == 0:
            self.logger.warning("Шортлист пуст для последней даты")
            return None
            
        # Добавляем колонку с рекомендацией
        shortlist_df['recommendation'] = 'HOLD'
        shortlist_df.loc[shortlist_df['signal'] == 1, 'recommendation'] = 'BUY'
        shortlist_df.loc[shortlist_df['signal'] == -1, 'recommendation'] = 'SELL'
        
        # Сортируем по композитному скору
        shortlist_df = shortlist_df.sort_values('composite_score', ascending=False)
        
        # Выбираем важные колонки для вывода
        if 'ticker' in shortlist_df.columns:
            output_columns = ['ticker', 'close', 'composite_score', 'tech_score', 
                              'sentiment_score', 'fundamental_score', 'signal', 'recommendation']
        else:
            output_columns = shortlist_df.columns
            
        output_columns = [col for col in output_columns if col in shortlist_df.columns]
        
        shortlists_dir = os.path.join(output_dir, 'shortlists')
        os.makedirs(shortlists_dir, exist_ok=True)
        
        # Сохраняем с датой в подпапку с историей
        dated_filename = os.path.join(shortlists_dir, f'shortlist_{latest_date}.csv')
        shortlist_df[output_columns].to_csv(dated_filename)
        self.logger.info(f"Шортлист сохранен в {dated_filename}")
        
        # Актуальную версию оставляем в корне для совместимости
        latest_filename = os.path.join(output_dir, 'results.csv')
        shortlist_df[output_columns].to_csv(latest_filename)
        self.logger.info(f"Актуальный шортлист сохранен в {latest_filename}")
            
        return shortlist_df[output_columns]
    
    def visualize_signals(self, output_dir=None):
        """
        Визуализирует график котировок с отмеченными сигналами покупки и продажи 
        для каждого тикера в отдельной папке
        """
        if self.df is None or 'signal' not in self.df.columns:
            self.logger.error("Нет данных с сигналами для визуализации")
            return None
            
        self.logger.info("Визуализация графиков котировок с торговыми сигналами")
        
        # Если директория для сохранения не указана, используем стандартную
        if output_dir is None:
            output_dir = 'signal_visualizations'
            
        # Создаем основную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # Проверяем наличие колонки ticker
        if 'ticker' not in self.df.columns:
            self.logger.error("Колонка 'ticker' не найдена, невозможно визуализировать по тикерам")
            return None
            
        # Получаем список уникальных тикеров
        tickers = self.df['ticker'].unique()
            
        for ticker in tickers:
            # Создаем папку для тикера
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Фильтруем данные для тикера
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                self.logger.warning(f"Нет данных для тикера {ticker}")
                continue
                
            # Создаем визуализацию сигналов
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # График цены закрытия
            ax.plot(ticker_data.index, ticker_data['close'], label='Цена закрытия', color='blue', alpha=0.7)
            
            # Отмечаем сигналы покупки (Buy)
            buy_signals = ticker_data[ticker_data['signal'] == 1]
            if len(buy_signals) > 0:
                ax.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, 
                           label='Покупка', zorder=5)
            
            # Отмечаем сигналы продажи (Sell)
            sell_signals = ticker_data[ticker_data['signal'] == -1]
            if len(sell_signals) > 0:
                ax.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, 
                           label='Продажа', zorder=5)
            
            # Настройка графика
            current_date = datetime.now().strftime('%Y-%m-%d')
            ax.set_title(f'График котировок и торговые сигналы: {ticker} (обновлено {current_date})')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Цена')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Форматирование дат на оси X
            plt.xticks(rotation=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.tight_layout()
            
            # Сохранение файла signals.png
            file_path = os.path.join(ticker_dir, 'signals.png')
            plt.savefig(file_path, dpi=300)
            plt.close()
            self.logger.info(f"График сигналов для {ticker} сохранен в {file_path}")
            saved_paths.append(file_path)
                
        return saved_paths
    
    def visualize_composite_scores(self, output_dir=None):
        """
        Визуализирует графики композитных скоров для каждого тикера в отдельной папке
        """
        if self.df is None or 'composite_score' not in self.df.columns:
            self.logger.error("Нет данных со скорами для визуализации")
            return None
            
        self.logger.info("Визуализация компонентов композитного скора")
        
        # Если директория для сохранения не указана, используем стандартную
        if output_dir is None:
            output_dir = 'signal_visualizations'
            
        # Создаем основную директорию
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # Проверяем наличие колонки ticker
        if 'ticker' not in self.df.columns:
            self.logger.error("Колонка 'ticker' не найдена, невозможно визуализировать по тикерам")
            return None
            
        # Получаем список уникальных тикеров
        tickers = self.df['ticker'].unique()
            
        for ticker in tickers:
            # Создаем папку для тикера
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Фильтруем данные для тикера
            ticker_data = self.df[self.df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                self.logger.warning(f"Нет данных для тикера {ticker}")
                continue
                
            # Создаем визуализацию с двумя подграфиками
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            # Верхний график: цена и сигналы
            ax1.plot(ticker_data.index, ticker_data['close'], label='Цена закрытия', color='blue', alpha=0.7)
            
            # Отмечаем сигналы покупки (Buy)
            buy_signals = ticker_data[ticker_data['signal'] == 1]
            if len(buy_signals) > 0:
                ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, 
                           label='Покупка', zorder=5)
            
            # Отмечаем сигналы продажи (Sell)
            sell_signals = ticker_data[ticker_data['signal'] == -1]
            if len(sell_signals) > 0:
                ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, 
                           label='Продажа', zorder=5)
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            ax1.set_title(f'Котировки и сигналы: {ticker} (обновлено {current_date})')
            ax1.set_ylabel('Цена')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Нижний график: композитный скор и его компоненты
            scores = ['composite_score']
            colors = ['black']
            
            if 'tech_score' in ticker_data.columns:
                scores.append('tech_score')
                colors.append('purple')
                
            if 'sentiment_score' in ticker_data.columns:
                scores.append('sentiment_score')
                colors.append('orange')
                
            if 'fundamental_score' in ticker_data.columns:
                scores.append('fundamental_score')
                colors.append('brown')
            
            for i, score in enumerate(scores):
                ax2.plot(ticker_data.index, ticker_data[score], label=score, color=colors[i])
            
            # Добавляем горизонтальные линии для пороговых значений сигналов
            ax2.axhline(y=self.threshold_buy, color='green', linestyle='--', alpha=0.7, 
                       label=f'Порог покупки ({self.threshold_buy})')
            ax2.axhline(y=self.threshold_sell, color='red', linestyle='--', alpha=0.7, 
                       label=f'Порог продажи ({self.threshold_sell})')
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            ax2.set_xlabel('Дата')
            ax2.set_ylabel('Скор')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
            # Форматирование дат
            plt.xticks(rotation=45)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.tight_layout()
            
            # Сохранение файла scores.png
            file_path = os.path.join(ticker_dir, 'scores.png')
            plt.savefig(file_path, dpi=300)
            plt.close()
            self.logger.info(f"График скоров для {ticker} сохранен в {file_path}")
            saved_paths.append(file_path)
                
        return saved_paths

    
    def run_pipeline(self, input_file=None, output_file=None, top_pct=0.3, output_dir=f'{BASE_PATH}/data/signal_visualizations'):
        """
        Запускает полный пайплайн генерации сигналов с сохранением результатов в структурированном виде
        
        Parameters:
        -----------
        input_file : str, optional
            Путь к CSV файлу с данными
        output_file : str, optional
            Путь для сохранения результатов
        top_pct : float
            Процент лучших акций для включения в shortlist
        output_dir : str
            Директория для сохранения визуализаций и результатов
            
        Returns:
        --------
        DataFrame с добавленными скорами и сигналами
        """
        self.logger.info("Запуск пайплайна генерации торговых сигналов")
        start_time = datetime.now()
        
        # Создаем основную директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Загрузка данных
        if input_file or self.df is None:
            self.load_data(input_file)
            
        if self.df is None:
            self.logger.error("Не удалось загрузить данные")
            return None
            
        # Расчет композитного скора
        self.calculate_composite_score()
        
        # Генерация сигналов
        self.generate_signals()
        
        # Создание shortlist
        self.create_shortlist(top_pct=top_pct)
        
        # Сохранение шортлиста
        self.save_shortlist(output_dir)
        
        # Визуализация сигналов
        self.visualize_signals(output_dir)
        
        # Визуализация компонентов скора
        self.visualize_composite_scores(output_dir)
        
        # Сохранение полных результатов
        today = datetime.now().strftime('%Y%m%d')

        # Создаем папку для истории сигналов
        signals_dir = os.path.join(output_dir, 'signals')
        os.makedirs(signals_dir, exist_ok=True)

        # Сохраняем историческую версию
        signals_historical_file = os.path.join(signals_dir, f'signals_{today}.csv')
        self.df.to_csv(signals_historical_file)
        self.logger.info(f"Исторические сигналы сохранены в {signals_historical_file}")

        # Сохраняем актуальную версию (либо в указанный файл, либо в корень)
        if output_file:
            self.df.to_csv(output_file)
            self.logger.info(f"Полные результаты сохранены в {output_file}")
        else:
            # Если output_file не указан, сохраняем в корне output_dir
            default_output = os.path.join(output_dir, 'signals_full.csv')
            self.df.to_csv(default_output)
            self.logger.info(f"Полные результаты сохранены в {default_output}")
        
        # Формируем сводку
        self.logger.info(f"Пайплайн выполнен за {(datetime.now() - start_time).total_seconds():.2f} секунд")
        self.logger.info(f"Обработано {len(self.df)} строк данных")
        if 'ticker' in self.df.columns:
            self.logger.info(f"Обработано {self.df['ticker'].nunique()} тикеров")
        
        self.logger.info(f"Создано сигналов: Buy - {(self.df['signal'] == 1).sum()}, " + 
                        f"Hold - {(self.df['signal'] == 0).sum()}, " + 
                        f"Sell - {(self.df['signal'] == -1).sum()}")
        
        if 'in_shortlist' in self.df.columns:
            self.logger.info(f"В шортлист включено {(self.df['in_shortlist']).sum()} записей")
            
        self.logger.info(f"Все результаты сохранены в каталоге: {output_dir}")
            
        return self.df
    
def run_pipeline_signal_generator(
        weight_tech=0.5,
        weight_sentiment=0.3,
        weight_fundamental=0.2
    ):

    SignalGenerator(
        input_file=f"{BASE_PATH}/data/df.csv",
        weight_tech=weight_tech,
        weight_sentiment=weight_sentiment,
        weight_fundamental=weight_fundamental
    ).run_pipeline(
        output_file=f"{BASE_PATH}/data/signals.csv",
        output_dir=f"{BASE_PATH}/data/signal_visualizations")
