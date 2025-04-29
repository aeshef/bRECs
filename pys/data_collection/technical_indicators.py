import pandas as pd
import matplotlib.pyplot as plt
import ta
import sys
import os
import logging

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class TechnicalIndicators(BaseLogger):
    def __init__(self, file_path, resample_rule="1D", sma_window=20, rsi_window=14):
        """
        Инициализация.
        :param file_path: путь к Parquet-файлу с минутными данными
        :param resample_rule: правило агрегации (например, "1D" для дневных данных)
        :param sma_window: окно для расчёта скользящей средней
        :param rsi_window: окно для расчёта RSI
        """
        super().__init__('TechnicalIndicators')
        self.file_path = file_path
        self.resample_rule = resample_rule
        self.sma_window = sma_window
        self.rsi_window = rsi_window
        self.df = None

    def load_data(self):
        """
        Загружает данные из Parquet-файла, преобразует временной столбец и
        агрегирует минутные данные до дневного масштаба с помощью resample_rule.
        Ожидается, что в файле имеются столбцы 'open', 'high', 'low', 'close', 'volume'
        и либо 'date', либо 'timestamp' (если timestamp в миллисекундах).
        :return: агрегированный DataFrame с дневными данными
        """
        self.logger.info("Начало загрузки данных.")
        self.df = pd.read_parquet(self.file_path)
        if 'min' in self.df.columns and 'max' in self.df.columns:
            self.df.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)

        print(self.df.columns)

        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        elif "timestamp" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["timestamp"], unit="ms")
        else:
            raise ValueError("Нет столбца 'date' или 'timestamp' в данных")
        self.df.set_index("date", inplace=True)
        self.df.sort_index(inplace=True)

        daily_df = (
            self.df.resample(self.resample_rule)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })
            .dropna()
        )
        self.df = daily_df
        return self.df

    def calculate_indicators(self):
        """
        Рассчитывает технические индикаторы по агрегированным дневным данным:

        - Скользящая средняя (SMA) и экспоненциальная скользящая средняя (EMA) по окну sma_window
        - Индекс относительной силы (RSI) по окну rsi_window
        - MACD, сигнальная линия и разница
        - Полосы Боллинджера (используется sma_window и стандартное отклонение=2)

        Returns
        -------
        pandas.DataFrame
            DataFrame с добавленными столбцами индикаторов.
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data().")
        
        self.logger.info("Начало расчёта технических индикаторов.")
        
        # Трендовые индикаторы
        self.df['SMA_20'] = self.df['close'].rolling(20).mean()
        self.df['EMA_20'] = self.df['close'].ewm(span=20, adjust=False).mean()
        self.df['WMA'] = self.df['close'].rolling(14).apply(lambda x: x[::-1].cumsum().sum()*2/(14*15), raw=True)
        
        # Индикаторы моментума
        self.df['RSI_14'] = ta.momentum.RSIIndicator(self.df['close'], 14).rsi()
        stoch = ta.momentum.StochasticOscillator(self.df['high'], self.df['low'], self.df['close'], 14)
        self.df['Stoch_%K'] = stoch.stoch()
        self.df['Stoch_%D'] = stoch.stoch_signal()
        
        # Волатильность
        bb = ta.volatility.BollingerBands(self.df['close'], 20, 2)
        self.df['BB_upper'] = bb.bollinger_hband()
        self.df['BB_mid'] = bb.bollinger_mavg()
        self.df['BB_lower'] = bb.bollinger_lband()
        self.df['ATR_14'] = ta.volatility.AverageTrueRange(self.df['high'], self.df['low'], self.df['close'], 14).average_true_range()
        
        # Объемные индикаторы
        self.df['OBV'] = ta.volume.OnBalanceVolumeIndicator(self.df['close'], self.df['volume']).on_balance_volume()
        self.df['CMF_20'] = ta.volume.ChaikinMoneyFlowIndicator(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], 20).chaikin_money_flow()
        
        # Трендовые системы
        macd = ta.trend.MACD(self.df['close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_signal'] = macd.macd_signal()
        self.df['MACD_diff'] = macd.macd_diff()
        
        # Дополнительные индикаторы
        self.df['ADX_14'] = ta.trend.ADXIndicator(self.df['high'], self.df['low'], self.df['close'], 14).adx()
        self.df['Vortex_14+'] = ta.trend.VortexIndicator(self.df['high'], self.df['low'], self.df['close'], 14).vortex_indicator_pos()
        self.df['Vortex_14-'] = ta.trend.VortexIndicator(self.df['high'], self.df['low'], self.df['close'], 14).vortex_indicator_neg()
        self.df['CCI_20'] = ta.trend.CCIIndicator(self.df['high'], self.df['low'], self.df['close'], 20).cci()
        self.df['Williams_%R_14'] = ta.momentum.WilliamsRIndicator(self.df['high'], self.df['low'], self.df['close'], 14).williams_r()
        
        # Ишимоку
        ichimoku = ta.trend.IchimokuIndicator(self.df['high'], self.df['low'], 9, 26)
        self.df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        self.df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        
        # Специальные индикаторы
        self.df['KAMA_10'] = ta.momentum.KAMAIndicator(self.df['close'], 10).kama()
        self.df['ROC_10'] = ta.momentum.ROCIndicator(self.df['close'], 10).roc()
        self.df['Ultimate_Osc'] = ta.momentum.UltimateOscillator(self.df['high'], self.df['low'], self.df['close']).ultimate_oscillator()
        
        return self.df
    
    def plot_price_with_indicators(self):
        """
        Строит график цены с наложенными скользящими средними и полосами Боллинджера.
        """
        if self.df is None or "SMA" not in self.df.columns:
            raise ValueError("Сначала загрузите данные и рассчитайте индикаторы (load_data() и calculate_indicators()).")
        
        self.logger.info("Построение графика с ценами")

        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df["close"], label="Close", color="blue")
        plt.plot(self.df.index, self.df["SMA_20"], label=f"SMA_20 ({self.sma_window})", color="orange")
        plt.plot(self.df.index, self.df["EMA_20"], label=f"EMA_20 ({self.sma_window})", color="green")
        plt.fill_between(self.df.index, self.df["BB_lower"], self.df["BB_upper"],
                         color="grey", alpha=0.3, label="Bollinger Bands")
        plt.xlabel("Дата")
        plt.ylabel("Цена")
        plt.title("Цена с техническими индикаторами (Дневные данные)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_macd(self):
        """
        Строит график MACD (линия MACD, сигнальная линия и разница).
        """
        if self.df is None or "MACD" not in self.df.columns:
            raise ValueError("Сначала загрузите данные и рассчитайте индикаторы (load_data() и calculate_indicators()).")
        
        self.logger.info("Построение MACD графика.")
        
        plt.figure(figsize=(14, 4))
        plt.plot(self.df.index, self.df["MACD"], label="MACD", color="blue")
        plt.plot(self.df.index, self.df["MACD_signal"], label="MACD Signal", color="red")
        plt.bar(self.df.index, self.df["MACD_diff"], label="MACD Diff", color="grey", alpha=0.5)
        plt.xlabel("Дата")
        plt.ylabel("Значение")
        plt.title("MACD (Дневные данные)")
        plt.legend()
        plt.grid(True)
        plt.show()