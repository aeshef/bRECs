import os
import re
import sys
import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# # Обеспечим, чтобы sys.stdout использовал UTF-8 (требуется для корректного вывода кириллических символов)
# sys.stdout.reconfigure(encoding='utf-8')

class SmartLabYearlyParser:
    def __init__(self, ticker, base_path='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
        """
        Инициализирует парсер для указанного тикера.
        
        Параметры:
          ticker    - тикер эмитента (например, "ROSN")
          base_path - базовый каталог для сохранения результатов
        """
        self.ticker = ticker.upper()
        self.url = f"https://smart-lab.ru/q/{self.ticker}/f/y/"
        self.base_path = base_path
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера с использованием UTF-8 для вывода."""
        # Настроим кастомный потоковой обработчик с кодировкой UTF-8:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger = logging.getLogger('SmartLabYearlyParser')
        logger.setLevel(logging.INFO)
        # Очистим предыдущие обработчики, если есть, и добавим наш
        logger.handlers.clear()
        logger.addHandler(handler)
        return logger

    def fetch_page(self):
        """
        Загружает HTML-страницу по URL для данного тикера.
        Возвращает:
            soup - объект BeautifulSoup для дальнейшего анализа или None при ошибке.
        """
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, как Gecko) "
                           "Chrome/115.0 Safari/537.36"),
            "Accept-Language": "ru-RU,ru;q=0.9"
        }
        self.logger.info(f"Запрос URL: {self.url}")
        try:
            response = requests.get(self.url, headers=headers)
            if response.status_code == 200:
                # Принудительно устанавливаем кодировку в UTF-8
                response.encoding = 'utf-8'
                self.logger.info("Успешное получение страницы.")
                html_content = response.text

                # Сохранить HTML для отладки (опционально)
                self._save_html_debug(html_content)

                soup = BeautifulSoup(html_content, "html.parser")
                return soup
            else:
                self.logger.error(f"Не удалось получить данные с {self.url}. HTTP Status Code: {response.status_code}")
                return None
        except Exception as e:
            # Принудительно преобразуем строку исключения в UTF-8-совместимый формат
            self.logger.error(f"Exception при запросе {self.url}: {str(e)}")
            return None

    def _save_html_debug(self, html):
        """
        Сохраняет полученный HTML для отладки.
        """
        debug_dir = os.path.join(self.base_path, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        filename = os.path.join(debug_dir, f"debug_{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
            self.logger.info(f"HTML для отладки сохранён: {filename}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения HTML для отладки: {str(e)}")

    def parse_yearly_tables(self, soup):
        """
        Извлекает таблицы с годовыми данными из объекта soup.
        Если найдено несколько таблиц, они объединяются в один DataFrame.
        Возвращает:
            DataFrame с объединёнными таблицами или пустой DataFrame, если ничего не найдено.
        """
        if soup is None:
            self.logger.error("Объект soup равен None. Прерывание парсинга таблиц.")
            return pd.DataFrame()

        tables = soup.find_all("table")
        if not tables:
            self.logger.error("На странице не найдены таблицы с годовыми данными.")
            return pd.DataFrame()

        self.logger.info(f"Найдено таблиц: {len(tables)}")
        dataframes = []
        for idx, table in enumerate(tables):
            table_html = str(table)
            try:
                # Читаем таблицу без установки заголовка, чтобы сохранить все строки
                df_list = pd.read_html(table_html, header=None, decimal=',')
                if df_list:
                    df = df_list[0]
                    # Добавляем информацию об источнике, если требуется
                    df["Источник"] = f"Таблица {idx+1}"
                    dataframes.append(df)
                    self.logger.info(f"Таблица {idx+1}: разобрана, строк - {len(df)}")
            except Exception as e:
                self.logger.warning(f"Таблица {idx+1} не может быть прочитана с помощью pd.read_html: {str(e)}")

        if dataframes:
            complete_df = pd.concat(dataframes, ignore_index=True, sort=False)
            self.logger.info(f"Объединённый DataFrame содержит {len(complete_df)} строк.")
            return complete_df
        else:
            self.logger.error("Не удалось извлечь ни одной таблицы.")
            return pd.DataFrame()

    def clean_fundamentals_table(self, df):
        """
        Обрабатывает полученный DataFrame с фундаментальными данными.
        Предполагается, что CSV имеет формат:
              Показатель,Показатель,Значение
        Выполняет действия:
          - Переименование первых трёх колонок в "Показатель1", "Показатель2", "Значение".
          - Объединяет колонки "Показатель1" и "Показатель2" в одну.
          - Удаляет пустые строки.
        Возвращает очищенный DataFrame с двумя столбцами: "Показатель" и "Значение".
        """
        df_clean = df.copy()
        # Приводим названия первых трёх колонок к строковому представлению
        col_names = list(df_clean.columns)
        col0 = str(col_names[0])
        col1 = str(col_names[1]) if len(col_names) > 1 else ""
        col2 = str(col_names[2]) if len(col_names) > 2 else ""
        
        # Если среди первых трёх колонок уже присутствуют ожидаемые названия, то переименуем их
        if ("Показатель" in col0) and ("Показатель" in col1) and ("Значение" in col2):
            df_clean.rename(columns={col_names[0]: "Показатель1", col_names[1]: "Показатель2", col_names[2]: "Значение"}, inplace=True)
        else:
            # Переименовываем независимо от исходных названий
            df_clean.rename(columns={col_names[0]: "Показатель1", col_names[1]: "Показатель2", col_names[2]: "Значение"}, inplace=True)

        # Объединяем две колонки показателей: если в первой ячейке есть значение, используем её, иначе – вторую
        df_clean["Показатель"] = df_clean.apply(lambda row: row["Показатель1"] if pd.notnull(row["Показатель1"]) and str(row["Показатель1"]).strip() != ""
                                                else row["Показатель2"], axis=1)
        # Оставляем только интересующие колонки
        df_clean = df_clean[["Показатель", "Значение"]]
        
        # Удаляем строки, где оба столбца пустые или содержат только пробелы
        df_clean = df_clean.dropna(how="all").applymap(lambda x: x if pd.isnull(x) else str(x).strip())
        df_clean = df_clean[(df_clean["Показатель"] != "") | (df_clean["Значение"] != "")]
        df_clean.reset_index(drop=True, inplace=True)
        return df_clean

    def extract_report_year(self, df):
        """
        Определяет год отчётности из строки с "Дата отчета".
        Если такая строка найдена, пытается распарсить дату в формате DD.MM.YYYY,
        затем возвращает год.
        Если не найдено или ошибка парсинга, возвращает "year".
        """
        report_year = "year"
        date_rows = df[df["Показатель"].str.contains("Дата отчета", case=False, na=False)]
        if not date_rows.empty:
            date_str = date_rows.iloc[0]["Значение"]
            try:
                report_date = datetime.strptime(date_str, "%d.%m.%Y")
                report_year = str(report_date.year)
            except Exception as e:
                self.logger.error(f"Ошибка парсинга даты '{date_str}': {str(e)}")
        else:
            self.logger.warning("Не найдена строка с 'Дата отчета'. Файл будет сохранён как 'year.csv'.")
        return report_year

    def process_and_save_fundamentals(self, df):
        """
        Обрабатывает объединённый DataFrame с фундаментальными данными, очищает его и сохраняет в CSV.
        Результат сохраняется по пути:
          {BASE_PATH}/{TICKER}/fundamental_data/{YEAR}.csv
        """
        if df.empty:
            self.logger.warning("DataFrame пустой, обработка прекращена.")
            return

        df_clean = self.clean_fundamentals_table(df)
        self.logger.info("Очистка таблицы фундаментальных показателей завершена.")

        if df_clean.empty:
            self.logger.error("После очистки таблицы нет данных.")
            return

        report_year = self.extract_report_year(df_clean)
        self.logger.info(f"Определён год отчёта: {report_year}")

        # Сохраняем CSV в каталоге:
        # Если годовые столбцы обнаружены, то сохранение происходит по годам, иначе агрегированный файл.
        folder_path = os.path.join(self.base_path, self.ticker, "fundamental_data")
        os.makedirs(folder_path, exist_ok=True)
        # Если report_year оказалось "year", сохраняем агрегированный файл
        if report_year == "year":
            file_path = os.path.join(folder_path, "aggregated.csv")
            try:
                df_clean.to_csv(file_path, index=False, encoding="utf-8")
                self.logger.info(f"Агрегированные фундаментальные данные сохранены в {file_path}.")
            except Exception as e:
                self.logger.error(f"Ошибка сохранения файла {file_path}: {str(e)}")
        else:
            # Сохраняем данные за этот год
            file_path = os.path.join(folder_path, f"{report_year}.csv")
            try:
                df_clean.to_csv(file_path, index=False, encoding="utf-8")
                self.logger.info(f"Сохранены фундаментальные данные за {report_year} в {file_path}.")
            except Exception as e:
                self.logger.error(f"Ошибка сохранения файла {file_path}: {str(e)}")
                
    def run(self):
        """
        Основной метод парсинга:
         – Загружает страницу.
         – Извлекает таблицы с данными.
         – Обрабатывает и сохраняет данные фундаментальных показателей.
        """
        self.logger.info(f"========== Обработка данных для тикера {self.ticker} ==========")
        soup = self.fetch_page()
        if soup is None:
            self.logger.error("Не удалось получить страницу. Парсинг прерван.")
            return

        df = self.parse_yearly_tables(soup)
        if df.empty:
            self.logger.warning("Нет данных для сохранения. Проверьте логи, HTML и структуру страницы.")
        else:
            self.process_and_save_fundamentals(df)

def run_pipeline(ticker_list, base_path='/Users/aeshef/Documents/GitHub/kursach/data/processed_data'):
    pipeline_logger = logging.getLogger('SmartLabYearlyPipeline')
    pipeline_logger.info(f"Запуск пайплайна для тикеров: {ticker_list}")
    for ticker in ticker_list:
        pipeline_logger.info(f"Начало обработки тикера: {ticker}")
        parser = SmartLabYearlyParser(ticker, base_path=base_path)
        parser.run()
        pipeline_logger.info(f"Завершена обработка тикера: {ticker}\n")