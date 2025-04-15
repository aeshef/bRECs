import pandas as pd
import matplotlib.pyplot as plt
import ta

class TechnicalIndicators:
    def __init__(self, file_path, resample_rule="1D", sma_window=20, rsi_window=14):
        """
        Инициализация.
        :param file_path: путь к Parquet-файлу с минутными данными
        :param resample_rule: правило агрегации (например, "1D" для дневных данных)
        :param sma_window: окно для расчёта скользящей средней
        :param rsi_window: окно для расчёта RSI
        """
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
        self.df = pd.read_parquet(self.file_path)

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
         • Скользящая средняя (SMA) и экспоненциальная скользящая средняя (EMA) по окну sma_window
         • Индекс относительной силы (RSI) по окну rsi_window
         • MACD, сигнальную линию и разницу
         • Полосы Боллинджера (с использованием sma_window и стандартного отклонения=2)
        :return: DataFrame с добавленными столбцами индикаторов
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data().")
        
        # Скользящие средние
        self.df["SMA"] = self.df["close"].rolling(window=self.sma_window).mean()
        self.df["EMA"] = self.df["close"].ewm(span=self.sma_window, adjust=False).mean()
        
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=self.df["close"], window=self.rsi_window)
        self.df["RSI"] = rsi_indicator.rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(close=self.df["close"])
        self.df["MACD"] = macd_indicator.macd()
        self.df["MACD_signal"] = macd_indicator.macd_signal()
        self.df["MACD_diff"] = macd_indicator.macd_diff()
        
        # Полосы Боллинджера
        bollinger = ta.volatility.BollingerBands(close=self.df["close"], window=self.sma_window, window_dev=2)
        self.df["BB_upper"] = bollinger.bollinger_hband()
        self.df["BB_lower"] = bollinger.bollinger_lband()
        self.df["BB_middle"] = bollinger.bollinger_mavg()
        
        return self.df

    def plot_price_with_indicators(self):
        """
        Строит график цены с наложенными скользящими средними и полосами Боллинджера.
        """
        if self.df is None or "SMA" not in self.df.columns:
            raise ValueError("Сначала загрузите данные и рассчитайте индикаторы (load_data() и calculate_indicators()).")
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df["close"], label="Close", color="blue")
        plt.plot(self.df.index, self.df["SMA"], label=f"SMA ({self.sma_window})", color="orange")
        plt.plot(self.df.index, self.df["EMA"], label=f"EMA ({self.sma_window})", color="green")
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