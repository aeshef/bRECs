import os
import requests
import pandas as pd
from pys.utils.logger import BaseLogger

class MarketCapParser(BaseLogger):
    """
    Класс для парсинга капитализаций компаний с Московской биржи
    и сохранения в структурированном виде
    """
    
    def __init__(self, base_path):
        """
        Инициализация парсера капитализаций
        
        :param base_path: Базовый путь для хранения данных
        """
        super().__init__('MarketCapParser')
        self.base_path = base_path
        self.processed_data_dir = os.path.join(self.base_path, "data", "processed_data")
    
    def _get_moex_market_caps(self, tickers=None):
        """
        Получает данные о капитализации компаний с Московской биржи
        
        :param tickers: Список тикеров для фильтрации (если None, возвращаются все тикеры)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame с тикерами и рыночной капитализацией
        """
        self.logger.info(f"Запрос данных о капитализации с MOEX API {'для выбранных тикеров' if tickers else 'для всех тикеров'}")
        
        # URL для получения данных по основному рынку акций
        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        
        # Параметры запроса
        params = {
            "iss.meta": "off",  # Отключаем мета-информацию
            "iss.only": "securities",  # Запрашиваем только секцию securities
            "securities.columns": "SECID,PREVPRICE,ISSUESIZE,ISSUECAPITALIZATION,MARKETVALUE"
        }
        
        # Выполняем запрос
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Проверяем на ошибки
            data = response.json()
        except Exception as e:
            self.logger.error(f"Ошибка при запросе данных: {e}")
            return pd.DataFrame(columns=["ticker", "market_cap"])
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(data["securities"]["data"], 
                         columns=data["securities"]["columns"])
        
        # Создаем итоговый DataFrame
        result_df = pd.DataFrame()
        result_df["ticker"] = df["SECID"]
        
        # Используем готовую капитализацию из API, если она есть
        if "MARKETVALUE" in df.columns and not df["MARKETVALUE"].isnull().all():
            result_df["market_cap"] = df["MARKETVALUE"]
        elif "ISSUECAPITALIZATION" in df.columns and not df["ISSUECAPITALIZATION"].isnull().all():
            result_df["market_cap"] = df["ISSUECAPITALIZATION"]
        else:
            # Рассчитываем капитализацию как цена * количество акций
            result_df["market_cap"] = df["PREVPRICE"] * df["ISSUESIZE"]
        
        # Удаляем строки с пустыми значениями капитализации
        result_df = result_df.dropna(subset=["market_cap"])
        
        # Фильтруем по списку тикеров, если он предоставлен
        if tickers:
            tickers_upper = [t.upper() for t in tickers]  # Переводим в верхний регистр для надежности
            result_df = result_df[result_df["ticker"].isin(tickers_upper)]
            
            # Проверяем, все ли запрошенные тикеры найдены
            found_tickers = set(result_df["ticker"].tolist())
            missing_tickers = set(tickers_upper) - found_tickers
            if missing_tickers:
                self.logger.warning(f"Не найдены данные о капитализации для следующих тикеров: {', '.join(missing_tickers)}")
        
        self.logger.info(f"Получены данные о капитализации для {len(result_df)} компаний")
        return result_df
    
    def _save_market_cap_for_ticker(self, ticker, market_cap):
        """
        Сохраняет данные о капитализации для конкретного тикера
        
        :param ticker: Тикер компании
        :param market_cap: Значение капитализации
        """
        # Создаем директорию для тикера
        ticker_dir = os.path.join(self.processed_data_dir, ticker, "market_cap")
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Путь к файлу с капитализацией
        cap_file = os.path.join(ticker_dir, "cap.csv")
        
        # Создаем и сохраняем DataFrame с капитализацией
        cap_df = pd.DataFrame({
            "ticker": [ticker],
            "market_cap": [market_cap]
        })
        
        cap_df.to_csv(cap_file, index=False)
        self.logger.debug(f"Капитализация для {ticker} сохранена в {cap_file}")
    
    def _save_full_market_caps(self, market_caps_df):
        """
        Сохраняет общий файл с капитализациями всех компаний
        
        :param market_caps_df: DataFrame с тикерами и капитализациями
        """
        data_dir = os.path.join(self.base_path, "data")
        market_cap_dir = os.path.join(data_dir)
        os.makedirs(market_cap_dir, exist_ok=True)
        
        cap_file = os.path.join(market_cap_dir, "all_caps.csv")
        
        market_caps_df.to_csv(cap_file, index=False)
        self.logger.info(f"Общий файл с капитализациями сохранен в {cap_file}")
    
    def run_pipeline_market_cap(self, tickers=None):
        """
        Запускает пайплайн парсинга капитализаций компаний.

        Args:
            tickers (list, optional): Список тикеров для получения капитализации (если None, запрашиваются все доступные).

        Returns:
            DataFrame: DataFrame с тикерами и рыночной капитализацией.
        """

        self.logger.info(f"Запуск пайплайна парсинга капитализаций {'для выбранных тикеров' if tickers else 'для всех компаний'}")
        
        market_caps_df = self._get_moex_market_caps(tickers)
        
        if market_caps_df.empty:
            self.logger.warning("Не удалось получить данные о капитализации")
            return market_caps_df
        
        for _, row in market_caps_df.iterrows():
            ticker = row["ticker"]
            market_cap = row["market_cap"]
            self._save_market_cap_for_ticker(ticker, market_cap)
        
        self._save_full_market_caps(market_caps_df)
        
        self.logger.info(f"Пайплайн парсинга капитализаций завершен успешно. Обработано {len(market_caps_df)} компаний.")
        return market_caps_df


def run_pipeline_market_cap(base_path, tickers=None):
    """
    Запускает пайплайн парсинга капитализаций компаний.

    Args:
        base_path (str): Базовый путь для хранения данных.
        tickers (list, optional): Список тикеров для получения капитализации (если None, запрашиваются все доступные).

    Returns:
        DataFrame: DataFrame с тикерами и рыночной капитализацией.
    """
    parser = MarketCapParser(base_path)
    return parser.run_pipeline_market_cap(tickers)