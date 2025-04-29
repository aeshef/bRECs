# /Твой_Проект_Kursach/core/pipeline_runner.py
import logging
import sys
import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Tuple
import json
import shutil
import base64
from io import BytesIO

# --- Импорт параметров из params.py ---
from .params import (
    COMMON_PARAMS, KBD_PARAMS, SIGNAL_PARAMS, PORTFOLIO_PARAMS,
    OPTIMIZATION_PARAMS, TICKERS, KBD_DATA_PATH, DATA_PATH, NEWS_PARAMS,
    PROJECT_ROOT, PYS_PATH, MARKET_DATA_PARAMS, INTEGRATION_PARAMS,
    REPORT_PARAMS, VISUALIZATION_PARAMS, BACKTEST_PARAMS,
    SELECT_PARAMS_BY_PROFILE, PORTFOLIO_CONTROLS, PORTFOLIO_TYPE_PARAMS
)

# --- Инициализация Кэша KBD ---
CACHED_KBD_DATA = None
LAST_KBD_UPDATE = None
KBD_CACHE_TTL = 7200  # 2 часа в секундах

# --- Настройка Логирования ---
logger = logging.getLogger(__name__) # Настройка должна быть в вызывающем коде (bot/scheduler)

# --- Загрузка .env (если еще не загружен) ---
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    logger.warning(f".env file not found at {env_path}")

# --- Получение переменных окружения ---
ENV_TYPE = os.getenv('ENV_TYPE', 'local')
MOCK_PIPELINES = os.getenv('MOCK_PIPELINES', 'False').lower() in ('true', '1', 't')
TINKOFF_TOKEN = os.getenv('TINKOFF_API_TOKEN')
TELEGRAM_API_ID = os.getenv('TELEGRAM_API_ID')
TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')

# --- Импорт Модулей Проекта ---
try:
    from pys.data_collection.private_info import BASE_PATH, token
    from pys.data_collection.market_data import run_pipeline_market
    from pys.data_collection.market_cap import run_pipeline_market_cap
    from pys.data_collection.fundamental_data import run_pipeline_fundamental
    from pys.data_collection.tech_analysis import run_pipeline_technical
    from pys.data_collection.kbd import run_pipeline_kbd_parser
    from pys.data_collection.news_pipeline import NewsPipeline
    from pys.data_collection.data_integration import run_pipeline_integration
    from pys.porfolio_optimization.executor import PipelineExecutor
except ImportError as e:
    logger.error(f"Could not import pipeline modules. Error: {e}")
    # Можно остановить или переключиться в MOCK
    MOCK_PIPELINES = True
    logger.warning("Switched to MOCK mode due to import errors.")

# --- Импорт Модулей Базы Данных ---
try:
    from db.models import SessionLocal, User, Portfolio, UserPreferences
    from db import crud
except ImportError as e:
     logger.critical(f"Fatal Error: Could not import database modules. Error: {e}")
     sys.exit(1)

# --- Константы для отчетов ---
REPORT_TYPES = {
    "portfolio_summary": "summary.md",
    "portfolio_weights": "weights.csv",
    "metrics_report": "metrics.json",
    "efficient_frontier": "efficient_frontier.png",
    "portfolio_pie": "portfolio_pie.png",
    "cumulative_returns": "cumulative_returns.png",
    "monthly_calendar": "monthly_returns.png",
    "drawdown_chart": "drawdown.png",
    "backtest_metrics": "backtest_metrics.txt"
}

# --- Функции для работы с KBD данными ---

def resolve_kbd_path() -> Path:
    """Разрешает путь к файлу KBD данных, учитывая относительную структуру проекта."""
    # Сначала пробуем относительный путь из params.py
    if KBD_DATA_PATH.exists():
        return KBD_DATA_PATH
    
    # Затем пробуем использовать BASE_PATH из pys.data_collection.private_info
    try:
        alt_path = Path(BASE_PATH) / "data" / "kbd" / "kbd_data.csv"
        if alt_path.exists():
            logger.info(f"Using alternative KBD path: {alt_path}")
            return alt_path
    except:
        pass
    
    # Наконец, пробуем найти в стандартных местах
    standard_paths = [
        PROJECT_ROOT / "data" / "kbd" / "kbd_data.csv",
        PROJECT_ROOT.parent / "data" / "kbd" / "kbd_data.csv",
        Path.home() / "kursach" / "data" / "kbd" / "kbd_data.csv"
    ]
    
    for path in standard_paths:
        if path.exists():
            logger.info(f"Found KBD data at standard path: {path}")
            return path
    
    # Если ничего не найдено, возвращаем путь по умолчанию для создания
    default_path = PROJECT_ROOT / "data" / "kbd" / "kbd_data.csv"
    logger.warning(f"No existing KBD data found. Will create at: {default_path}")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


def get_cached_kbd_data() -> pd.DataFrame:
    """Возвращает кэшированные данные KBD с автоматическим обновлением."""
    global CACHED_KBD_DATA, LAST_KBD_UPDATE
    
    now = datetime.now()
    if CACHED_KBD_DATA is not None and LAST_KBD_UPDATE is not None:
        time_diff = (now - LAST_KBD_UPDATE).total_seconds()
        if time_diff < KBD_CACHE_TTL:
            logger.debug("Using cached KBD data.")
            return CACHED_KBD_DATA

    # Найдем правильный путь к файлу KBD данных
    kbd_path = resolve_kbd_path()
    logger.info(f"Checking KBD data file at: {kbd_path}")
    
    if kbd_path.exists():
        try:
            # Проверим возраст файла, чтобы не читать слишком старый кэш
            file_mod_time = datetime.fromtimestamp(kbd_path.stat().st_mtime)
            if (now - file_mod_time).total_seconds() < KBD_CACHE_TTL * 1.5: # Даем запас
                 logger.info("Loading KBD data from file cache...")
                 CACHED_KBD_DATA = pd.read_csv(kbd_path)
                 LAST_KBD_UPDATE = now
                 logger.info(f"Loaded {len(CACHED_KBD_DATA)} rows from {kbd_path}")
                 return CACHED_KBD_DATA
            else:
                 logger.warning("KBD file cache is older than TTL. Refreshing.")
        except Exception as e:
            logger.error(f"Error loading KBD data from file cache: {e}")

    logger.info("No valid KBD cache found. Running KBD pipeline...")
    try:
        # Убедимся, что директория существует
        kbd_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Убедимся, что start_date и end_date передаются правильно
        start_date_kbd = datetime.strptime(COMMON_PARAMS['start_date'], '%Y-%m-%d')
        end_date_kbd = datetime.strptime(COMMON_PARAMS['end_date'], '%Y-%m-%d')

        # Убедимся, что функция получает правильный base_path
        try:
            base_path_for_kbd = BASE_PATH  # Из private_info
        except:
            base_path_for_kbd = str(PROJECT_ROOT)
        
        # Запускаем пайплайн KBD
        run_pipeline_kbd_parser(
            base_path=base_path_for_kbd,
            start_date=start_date_kbd.strftime('%Y-%m-%d'), 
            end_date=end_date_kbd.strftime('%Y-%m-%d'),
            update_data=True
        )
        
        # Проверяем, создался ли файл
        if not kbd_path.exists():
             logger.error(f"KBD data file was not created at {kbd_path} after pipeline run!")
             # Попробуем найти файл в других местах
             alternative_paths = [
                 Path(base_path_for_kbd) / "data" / "kbd" / "kbd_data.csv",
                 Path("data") / "kbd" / "kbd_data.csv"
             ]
             for alt_path in alternative_paths:
                 if alt_path.exists():
                     logger.info(f"Found KBD data at alternative location: {alt_path}")
                     # Копируем файл в ожидаемое место
                     shutil.copy2(alt_path, kbd_path)
                     break
        
        # Если после всех попыток файл существует, загружаем его
        if kbd_path.exists():
            CACHED_KBD_DATA = pd.read_csv(kbd_path)
            LAST_KBD_UPDATE = now
            logger.info(f"Successfully updated KBD data: {len(CACHED_KBD_DATA)} rows saved to {kbd_path}")
        else:
            logger.error("Failed to locate KBD data file after pipeline run.")
            CACHED_KBD_DATA = pd.DataFrame()
            
    except Exception as e:
        logger.exception(f"Failed to run KBD pipeline or load data after update: {e}")
        CACHED_KBD_DATA = pd.DataFrame()
    
    return CACHED_KBD_DATA


def run_global_data_update() -> bool:
    """Запускает все пайплайны сбора и предобработки данных, используя параметры из params.py."""
    logger.info("Starting global data update using parameters from params.py...")

    if not TINKOFF_TOKEN:
        logger.error("Tinkoff API token not found. Cannot run market data pipeline.")
        return False

    try:
        # Получаем даты из COMMON_PARAMS
        start_date = datetime.strptime(COMMON_PARAMS['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(COMMON_PARAMS['end_date'], '%Y-%m-%d')
        # Можно также использовать динамические даты для некоторых пайплайнов
        now = datetime.now()
        start_date_short = now - timedelta(days=10)
        start_date_long = now - timedelta(days=400)

        try:
            # Убедимся, что можем получить BASE_PATH из private_info
            base_path = BASE_PATH
            token_value = token if token else TINKOFF_TOKEN
        except:
            base_path = str(PROJECT_ROOT)
            token_value = TINKOFF_TOKEN
            
        processed_data_path = Path(base_path) / "data" / "processed_data"
        processed_data_path.mkdir(parents=True, exist_ok=True)

        # 1. Market Data
        logger.info("Running market data pipeline...")
        run_pipeline_market(
            tickers=TICKERS,
            start_date=COMMON_PARAMS['start_date'],
            end_date=COMMON_PARAMS['end_date'],
            token=token_value,
            timeframe=MARKET_DATA_PARAMS['timeframe']
        )
        logger.info("Market data pipeline finished.")

        # 2. Market Cap
        logger.info("Running market cap pipeline...")
        run_pipeline_market_cap(
            base_path=base_path,
            tickers=TICKERS
        )
        logger.info("Market cap pipeline finished.")

        # 3. Fundamental Data
        logger.info("Running fundamental data pipeline...")
        run_pipeline_fundamental(
            ticker_list=TICKERS,
            base_path=base_path
        )
        logger.info("Fundamental data pipeline finished.")

        # 4. Technical Analysis
        logger.info("Running technical analysis pipeline...")
        run_pipeline_technical(
            tickers=TICKERS,
            base_dir=base_path
        )
        logger.info("Technical analysis pipeline finished.")

        # 5. KBD Data
        logger.info("Initializing KBD data (will run pipeline if needed)...")
        kbd_data = get_cached_kbd_data() # Используем кэширующую функцию
        if kbd_data.empty:
            logger.error("Critical error: KBD data initialization failed! Stopping global update.")
            return False
        logger.info(f"KBD data initialized ({len(kbd_data)} records).")

        # 6. News Data
        logger.info("Running news pipeline...")
        # Конвертируем строки дат в объекты date для функции
        news_start_date = datetime.strptime(NEWS_PARAMS['start_date'], '%Y-%m-%d').date()
        news_end_date = datetime.strptime(NEWS_PARAMS['end_date'], '%Y-%m-%d').date()
        
        # Собираем параметры для вызова, исключая start_date/end_date из NEWS_PARAMS
        news_call_params = {k: v for k, v in NEWS_PARAMS.items() if k not in ['start_date', 'end_date']}

        if TELEGRAM_API_ID and TELEGRAM_API_HASH:
            NewsPipeline.run_pipeline(
                base_dir=base_path,
                tickers=TICKERS,
                collect_telegram=True,
                telegram_api_id=int(TELEGRAM_API_ID),
                telegram_api_hash=TELEGRAM_API_HASH,
                start_date=news_start_date,
                end_date=news_end_date,
                **news_call_params
            )
            logger.info("News pipeline finished (with Telegram).")
        else:
            logger.warning("Telegram API ID/Hash not configured. Running news pipeline without Telegram.")
            NewsPipeline.run_pipeline(
                base_dir=base_path,
                tickers=TICKERS,
                collect_telegram=False,
                start_date=news_start_date,
                end_date=news_end_date,
                **news_call_params
            )
            logger.info("News pipeline finished (without Telegram).")

        # 7. Data Integration
        logger.info("Running data integration pipeline...")
        output_integration_path = DATA_PATH / "df.csv"
        run_pipeline_integration(
            tickers=TICKERS,
            output_path=str(output_integration_path),
            method=INTEGRATION_PARAMS['method']
        )
        logger.info(f"Data integration pipeline finished. Output: {output_integration_path}")

        logger.info("Global data update finished successfully.")
        return True

    except Exception as e:
        logger.exception(f"Error during real global data update: {e}")
        return False


# --- Функции для формирования отчетов ---

def collect_portfolio_report_files(pipeline_path: Path) -> Dict[str, Optional[Path]]:
    """
    Собирает файлы отчетов из директории выходных данных пайплайна.
    
    Args:
        pipeline_path: Путь к директории с результатами пайплайна
        
    Returns:
        Словарь с типами отчетов и путями к файлам
    """
    report_files = {}
    
    # 1. Проверяем основную директорию
    for report_type, filename in REPORT_TYPES.items():
        file_path = pipeline_path / filename
        if file_path.exists():
            report_files[report_type] = file_path
        else:
            # Если не нашли в корне, ищем глубже
            for subdir in pipeline_path.glob('**/'):
                subfile_path = subdir / filename
                if subfile_path.exists():
                    report_files[report_type] = subfile_path
                    break
    
    # 2. Проверяем директорию final_portfolio
    final_dir = pipeline_path / "final_portfolio"
    if final_dir.exists():
        for report_type, filename in REPORT_TYPES.items():
            if report_type not in report_files:
                file_path = final_dir / filename
                if file_path.exists():
                    report_files[report_type] = file_path
    
    # 3. Ищем другие полезные файлы
    for subdir in pipeline_path.glob('**/'):
        for chart_file in subdir.glob('*.png'):
            if chart_file.name.startswith('portfolio') and 'portfolio_pie' not in report_files:
                report_files['portfolio_pie'] = chart_file
            if chart_file.name.startswith('combined_portfolio_pie') and 'combined_pie' not in report_files:
                report_files['combined_pie'] = chart_file
            if chart_file.name.startswith('cumulative') and 'cumulative_returns' not in report_files:
                report_files['cumulative_returns'] = chart_file
        
        # Ищем текстовые отчеты
        for text_file in subdir.glob('*summary*.txt'):
            if 'portfolio_summary_text' not in report_files:
                report_files['portfolio_summary_text'] = text_file
                
        for text_file in subdir.glob('*metrics*.txt'):
            if 'metrics_text' not in report_files:
                report_files['metrics_text'] = text_file
    
    return report_files


def extract_portfolio_data(run_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Извлекает данные о портфеле (веса и метрики) из файлов отчетов.
    
    Args:
        run_path: Путь к директории с результатами запуска
        
    Returns:
        Кортеж (weights, metrics)
    """
    weights = {}
    metrics = {}
    
    # Собираем файлы отчетов
    report_files = collect_portfolio_report_files(run_path)
    
    # Извлекаем веса из CSV файла
    if 'portfolio_weights' in report_files:
        try:
            weights_df = pd.read_csv(report_files['portfolio_weights'])
            
            if 'Ticker' in weights_df.columns and 'Weight' in weights_df.columns:
                for _, row in weights_df.iterrows():
                    ticker = row['Ticker']
                    weight = float(row['Weight'])
                    weights[ticker] = weight
            elif len(weights_df.columns) >= 2:  # Предполагаем, что первая колонка - тикер, вторая - вес
                for _, row in weights_df.iterrows():
                    ticker = row.iloc[0]
                    weight = float(row.iloc[1])
                    weights[ticker] = weight
        except Exception as e:
            logger.error(f"Error extracting weights from CSV: {e}")
    
    # Если не нашли CSV, ищем в JSON
    if not weights and 'metrics_report' in report_files:
        try:
            with open(report_files['metrics_report'], 'r') as f:
                metrics_data = json.load(f)
                if 'weights' in metrics_data:
                    weights = metrics_data['weights']
        except Exception as e:
            logger.error(f"Error extracting weights from JSON: {e}")
    
    # Извлекаем метрики из JSON или текстового файла
    if 'metrics_report' in report_files:
        try:
            with open(report_files['metrics_report'], 'r') as f:
                metrics_data = json.load(f)
                # Очищаем от weights для метрик
                if 'weights' in metrics_data:
                    metrics_data.pop('weights')
                metrics = metrics_data
        except Exception as e:
            logger.error(f"Error extracting metrics from JSON: {e}")
    
    # Если не нашли в JSON, пробуем текстовый файл
    if not metrics and 'metrics_text' in report_files:
        try:
            metrics = {}
            with open(report_files['metrics_text'], 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value_str = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value_str = value_str.strip()
                        # Попробуем извлечь числовое значение
                        if '%' in value_str:
                            value_str = value_str.replace('%', '')
                            try:
                                metrics[key] = float(value_str) / 100.0
                            except:
                                pass
                        else:
                            try:
                                metrics[key] = float(value_str)
                            except:
                                pass
        except Exception as e:
            logger.error(f"Error extracting metrics from text: {e}")
    
    # Если всё ещё нет метрик, ищем в README.md
    if not metrics and (run_path / 'final_portfolio' / 'README.md').exists():
        try:
            with open(run_path / 'final_portfolio' / 'README.md', 'r') as f:
                readme_text = f.read()
                # Ищем строки с метриками в формате "- Ожидаемая доходность: XX.XX%"
                metric_lines = [line for line in readme_text.split('\n') if line.startswith('- ') and ':' in line]
                for line in metric_lines:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].replace('-', '').strip().lower().replace(' ', '_')
                        value_str = parts[1].strip()
                        # Попробуем извлечь числовое значение
                        if '%' in value_str:
                            value_str = value_str.replace('%', '')
                            try:
                                metrics[key] = float(value_str) / 100.0
                            except:
                                pass
                        else:
                            try:
                                metrics[key] = float(value_str)
                            except:
                                pass
        except Exception as e:
            logger.error(f"Error extracting metrics from README: {e}")
    
    return weights, metrics


def get_report_images(run_path: Path) -> Dict[str, str]:
    """
    Получает пути к графическим отчетам и конвертирует их в строки base64.
    
    Args:
        run_path: Путь к директории с результатами запуска
        
    Returns:
        Словарь с типами графиков и их данными в формате base64
    """
    image_data = {}
    report_files = collect_portfolio_report_files(run_path)
    
    # Извлекаем изображения
    for report_type, file_path in report_files.items():
        if file_path and file_path.exists() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            try:
                with open(file_path, 'rb') as img_file:
                    encoded = base64.b64encode(img_file.read()).decode('utf-8')
                    image_data[report_type] = encoded
            except Exception as e:
                logger.error(f"Error encoding image {file_path}: {e}")
    
    return image_data


def format_portfolio_summary(weights: Dict[str, float], metrics: Dict[str, float]) -> str:
    """
    Форматирует данные о портфеле в читаемый текстовый отчет.
    
    Args:
        weights: Словарь с весами активов
        metrics: Словарь с метриками портфеля
        
    Returns:
        Отформатированный текстовый отчет
    """
    # Форматируем отчет в MarkDown формате
    output = "```\n"
    output += "╔═══════════════════════════════════════════════════╗\n"
    output += "║                ИНВЕСТИЦИОННЫЙ ПОРТФЕЛЬ             ║\n"
    output += "╚═══════════════════════════════════════════════════╝\n\n"
    
    # Метрики портфеля
    output += "┌─────────────── КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ───────────────┐\n"
    
    # Преобразуем метрики и форматируем их
    metric_labels = {
        'expected_return': 'Ожидаемая доходность',
        'expected_volatility': 'Ожидаемая волатильность',
        'sharpe_ratio': 'Коэффициент Шарпа',
        'annual_return': 'Годовая доходность',
        'annual_volatility': 'Годовая волатильность',
        'max_drawdown': 'Максимальная просадка',
        'win_rate': 'Процент выигрышных периодов',
        'sortino_ratio': 'Коэффициент Сортино',
        'calmar_ratio': 'Коэффициент Калмара',
        'ожидаемая_доходность': 'Ожидаемая доходность',
        'ожидаемая_волатильность': 'Ожидаемая волатильность',
        'коэффициент_шарпа': 'Коэффициент Шарпа'
    }
    
    # Форматируем метрики
     # Форматируем метрики
    for key, label in metric_labels.items():
        if key in metrics:
            value = metrics[key]
            if key in ['expected_return', 'expected_volatility', 'annual_return', 'annual_volatility', 'max_drawdown', 'win_rate']:
                output += f"│ {label:30} │ {value*100:7.2f}% │\n"  # Проверьте форматирование
            else:
                output += f"│ {label:30} │ {value:7.2f}  │\n"
    
    output += "└───────────────────────────────────────────────────┘\n\n"
    
    # Разделение активов на группы
    stocks = {}
    bonds = {}
    shorts = {}
    other = {}
    
    for ticker, weight in weights.items():
        if ticker == 'RISK_FREE':
            continue  # Пропускаем RISK_FREE для отдельной секции
        if 'ОФЗ' in ticker or 'OFZ' in ticker or ticker.startswith('RU000') or 'Bond' in ticker:
            bonds[ticker] = weight
        elif weight < 0:
            shorts[ticker] = weight
        elif weight > 0:
            stocks[ticker] = weight
        else:
            other[ticker] = weight
    
    # Доля безрисковых активов
    risk_free_weight = weights.get('RISK_FREE', 0)
    
    # Распределение активов
    output += "┌──────────── РАСПРЕДЕЛЕНИЕ ПОРТФЕЛЯ ────────────┐\n"
    output += f"│ Безрисковые активы (облигации): {risk_free_weight*100:7.2f}%    │\n"
    output += f"│ Акции (длинные позиции):       {sum(stocks.values())*100:7.2f}%    │\n"
    
    if shorts:
        output += f"│ Акции (короткие позиции):      {sum(shorts.values())*100:7.2f}%    │\n"
    
    if bonds and bonds != {'RISK_FREE': risk_free_weight}:
        output += f"│ Дополнительные облигации:      {sum(bonds.values())*100:7.2f}%    │\n"
        
    output += "└───────────────────────────────────────────────┘\n\n"
    
    # Состав портфеля
    output += "┌─────────────── СОСТАВ ПОРТФЕЛЯ ───────────────┐\n"
    output += "│ Тикер            │  Вес   │        Тип        │\n"
    output += "├──────────────────┼────────┼───────────────────┤\n"
    
    # Сначала выводим RISK_FREE
    if risk_free_weight > 0:
        output += f"│ RISK_FREE        │ {risk_free_weight*100:6.2f}% │ Безрисковые активы │\n"
    
    # Затем сортированные по весу акции (длинные позиции)
    for ticker, weight in sorted(stocks.items(), key=lambda x: x[1], reverse=True):
        type_label = "Акции (длинные)"
        output += f"│ {ticker:16} │ {weight*100:6.2f}% │ {type_label:17} │\n"
    
    # Затем короткие позиции
    for ticker, weight in sorted(shorts.items(), key=lambda x: x[1]):
        type_label = "Акции (короткие)"
        output += f"│ {ticker:16} │ {weight*100:6.2f}% │ {type_label:17} │\n"
    
    # Затем облигации (кроме RISK_FREE)
    for ticker, weight in sorted(bonds.items(), key=lambda x: x[1], reverse=True):
        type_label = "Облигации"
        output += f"│ {ticker:16} │ {weight*100:6.2f}% │ {type_label:17} │\n"
    
    # Затем прочие активы
    for ticker, weight in sorted(other.items(), key=lambda x: x[1], reverse=True):
        type_label = "Прочие активы"
        output += f"│ {ticker:16} │ {weight*100:6.2f}% │ {type_label:17} │\n"
    
    output += "└──────────────────┴────────┴───────────────────┘\n"
    output += "```"
    
    return output


def generate_full_report(run_path: Path) -> Dict[str, Any]:
    """
    Генерирует полный отчет о портфеле на основе результатов запуска пайплайна.
    
    Args:
        run_path: Путь к директории с результатами запуска
        
    Returns:
        Словарь с данными отчета
    """
    # Извлекаем данные о портфеле
    weights, metrics = extract_portfolio_data(run_path)
    
    # Получаем графики в формате base64
    images = get_report_images(run_path)
    
    # Форматируем текстовый отчет
    text_report = format_portfolio_summary(weights, metrics)
    
    # Определяем тип портфеля на основе данных
    portfolio_type = "balanced"  # по умолчанию
    if any(weight < 0 for weight in weights.values()):
        if sum(weight for weight in weights.values() if weight > 0) > 0.8:
            portfolio_type = "combined"
        else:
            portfolio_type = "short"
    elif weights.get('RISK_FREE', 0) > 0.6:
        portfolio_type = "conservative"
    elif weights.get('RISK_FREE', 0) < 0.3:
        portfolio_type = "aggressive"
    
    # Формируем рекомендации на основе типа портфеля и метрик
    recommendations = generate_recommendations(portfolio_type, metrics, weights)
    
    # Объединяем все в один отчет
    report = {
        "weights": weights,
        "metrics": metrics,
        "images": images,
        "text_report": text_report,
        "portfolio_type": portfolio_type,
        "recommendations": recommendations,
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report


def generate_recommendations(portfolio_type: str, metrics: Dict[str, float], weights: Dict[str, float]) -> List[str]:
    """
    Генерирует рекомендации на основе типа портфеля и его показателей.
    
    Args:
        portfolio_type: Тип портфеля (conservative, balanced, aggressive, short, combined)
        metrics: Метрики портфеля
        weights: Веса активов в портфеле
        
    Returns:
        Список рекомендаций
    """
    recommendations = []
    
    # Базовые рекомендации в зависимости от типа портфеля
    if portfolio_type == "conservative":
        recommendations.append("Этот консервативный портфель подходит для сохранения капитала с умеренным ростом.")
        recommendations.append("Рекомендуется для горизонта инвестирования от 1 года.")
    elif portfolio_type == "balanced":
        recommendations.append("Сбалансированный портфель предлагает оптимальное соотношение риска и доходности.")
        recommendations.append("Рекомендуется для среднесрочного инвестирования (от 1 до 3 лет).")
    elif portfolio_type == "aggressive":
        recommendations.append("Агрессивный портфель нацелен на максимальную доходность при повышенном риске.")
        recommendations.append("Рекомендуется для долгосрочного инвестирования (от 3 лет) с толерантностью к риску.")
    elif portfolio_type == "short":
        recommendations.append("Портфель включает короткие позиции (шорт) для получения дохода на падающем рынке.")
        recommendations.append("Требует активного управления и мониторинга. Подходит для опытных инвесторов.")
    elif portfolio_type == "combined":
        recommendations.append("Комбинированный портфель сочетает длинные и короткие позиции для улучшения соотношения риск-доходность.")
        recommendations.append("Рекомендуется для инвесторов, стремящихся к рыночно-нейтральным стратегиям.")
    
    # Дополнительные рекомендации на основе метрик
    expected_return = metrics.get('expected_return', metrics.get('annual_return', 0))
    volatility = metrics.get('expected_volatility', metrics.get('annual_volatility', 0))
    sharpe = metrics.get('sharpe_ratio', 0)
    max_drawdown = metrics.get('max_drawdown', 0)
    
    if expected_return < 0.05:
        recommendations.append("Портфель имеет относительно низкую ожидаемую доходность. Рассмотрите увеличение доли активов роста для долгосрочных целей.")
    
    if volatility > 0.25:
        recommendations.append("Портфель имеет высокую волатильность. Рассмотрите увеличение доли облигаций для снижения рисков.")
    
    if sharpe < 0.5:
        recommendations.append("Низкий коэффициент Шарпа указывает на недостаточную компенсацию за принимаемый риск. Рекомендуется пересмотр активов.")
    elif sharpe > 1.0:
        recommendations.append("Высокий коэффициент Шарпа свидетельствует о хорошем соотношении риск-доходность. Оптимально для текущих условий.")
    
    if max_drawdown > 0.3:
        recommendations.append("Портфель демонстрирует высокую просадку. Для снижения просадок рекомендуется увеличить диверсификацию.")
    
    # Рекомендации по оптимальному количеству активов
    risk_free_weight = weights.get('RISK_FREE', 0)
    stock_count = sum(1 for ticker, weight in weights.items() if weight > 0 and ticker != 'RISK_FREE')
    
    if stock_count < 5 and risk_free_weight < 0.7:
        recommendations.append("Портфель содержит небольшое количество акций. Рекомендуется увеличить диверсификацию.")
    elif stock_count > 15:
        recommendations.append("Портфель высоко диверсифицирован. Это снижает потенциал как больших потерь, так и значительного роста.")
    
    # Общие рекомендации
    recommendations.append("Регулярно пересматривайте состав портфеля (не реже раза в квартал).")
    
    if portfolio_type in ["short", "combined"]:
        recommendations.append("Необходимо внимательно следить за стоимостью заимствований при коротких позициях.")
    
    return recommendations


def _save_portfolio_results(user_id: int, weights: dict, metrics: dict, strategy_profile: str, pipeline_name: str) -> bool:
    """Внутренняя функция для сохранения портфеля в БД."""
    db_save = SessionLocal()
    try:
        new_portfolio = crud.create_portfolio(
            db=db_save,
            user_id=user_id,
            name=pipeline_name,
            weights=weights,
            metrics=metrics,
            strategy_profile=strategy_profile
        )
        logger.info(f"Successfully saved new portfolio (ID: {new_portfolio.id}) for user {user_id}.")
        return True
    except Exception as save_err:
        logger.exception(f"Database error saving portfolio for user {user_id}: {save_err}")
        return False
    finally:
        db_save.close()


# --- Обновление Портфеля Пользователя ---
def run_user_portfolio_update(user_id: int, is_initial: bool = False) -> Optional[Tuple[Optional[Dict], Optional[Dict], bool, Optional[Dict]]]:
    """
    Запускает пайплайн генерации портфеля для пользователя.
    
    Args:
        user_id: ID пользователя
        is_initial: Флаг первичного создания портфеля
        
    Returns:
        Кортеж (dict_весов_или_None, dict_метрик_или_None, bool_значительные_изменения, dict_отчета_или_None)
    """
    operation = "Initial generation" if is_initial else "Update"
    logger.info(f"Running portfolio {operation.lower()} for user_id: {user_id}")

    # --- 1. Получение данных пользователя из БД ---
    db = SessionLocal()
    preferences = None
    strategy_profile = None
    
    try:
        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user:
            logger.warning(f"User {user_id} not found in DB. Skipping.")
            return None, None, False, None
        if not db_user.is_active:
            logger.warning(f"User {user_id} is not active. Skipping.")
            return None, None, False, None

        # Получаем или устанавливаем риск-профиль
        strategy_profile = db_user.risk_profile
        if not strategy_profile:
            logger.warning(f"User {user_id} has no risk profile. Assigning 'moderate'.")
            try:
                db_user = crud.update_user_risk_profile(db, db_user.telegram_id, 'moderate')
                strategy_profile = 'moderate'
                if not db_user: raise Exception("Failed to assign default risk profile.")
            except Exception as profile_err:
                logger.error(f"Could not assign default profile to user {user_id}: {profile_err}")
                return None, None, False, None

        # Получаем или создаем предпочтения
        preferences = crud.get_user_preferences(db, user_id)
        if not preferences:
            logger.warning(f"User {user_id} preferences not found. Creating defaults.")
            try:
                # Получаем дефолтное allow_short из params.py на основе профиля
                default_allow_short = PORTFOLIO_PARAMS.get(strategy_profile, {}).get('allow_short', False)
                preferences = crud.create_or_update_preferences(db, user_id, allow_short=default_allow_short)
                if not preferences: raise Exception("Failed to create default preferences.")
                logger.info(f"Created default preferences for user {user_id} with allow_short={default_allow_short}")
            except Exception as prefs_err:
                logger.error(f"Could not create default preferences for user {user_id}: {prefs_err}")
                return None, None, False, None

        logger.info(f"User {user_id}: Strategy profile = '{strategy_profile}', Allow Short = {preferences.allow_short}")

    except Exception as db_err:
        logger.exception(f"Database error fetching user {user_id} data: {db_err}")
        return None, None, False, None
    finally:
        db.close() # Закрываем сессию

    # --- 2. MOCK Режим (если включен) ---
    if MOCK_PIPELINES:
        logger.warning(f"[MOCK] Simulating portfolio {operation.lower()} for user {user_id} ({strategy_profile}).")
        time.sleep(random.uniform(5, 10)) # Имитация

        # Логика генерации мок-портфеля
        mock_weights = {}
        mock_tickers = ['SBER', 'GAZP', 'LKOH', 'NVTK', 'YNDX', 'POLY', 'PLZL', 'MGNT', 'TATN', 'ROSN']
        num_stocks = random.randint(preferences.max_stocks // 2, preferences.max_stocks)
        num_bonds = random.randint(preferences.max_bonds // 2, preferences.max_bonds)
        sampled_stocks = random.sample(mock_tickers, num_stocks)
        bond_tickers = [f"OFZ-{random.randint(26200, 26240)}" for _ in range(num_bonds)]

        # РАСЧЕТ ВЕСОВ (упрощенный, как в старом коде)
        total_weight = 0
        base_params = PORTFOLIO_PARAMS.get(strategy_profile, PORTFOLIO_PARAMS['moderate']) # Параметры для профиля

        # Доли облигаций
        target_bond_weight = random.uniform(base_params['min_rf_allocation'], base_params['max_rf_allocation'])
        current_bond_weight = 0
        bond_weights_raw = {}
        for ticker in bond_tickers:
            w_bond = random.uniform(0.03, 0.1)
            bond_weights_raw[ticker] = w_bond
            current_bond_weight += w_bond
        # Масштабируем облигации
        if current_bond_weight > 0:
            scale_factor_bonds = target_bond_weight / current_bond_weight
            for ticker, w_bond in bond_weights_raw.items():
                scaled_w = w_bond * scale_factor_bonds
                mock_weights[ticker] = scaled_w
                total_weight += scaled_w

        # Доли акций (остаток)
        stock_weight_target = 1.0 - sum(mock_weights.get(bt, 0) for bt in bond_tickers)
        current_stock_weight = 0
        stock_weights_raw = {}
        for ticker in sampled_stocks:
            w_stock = random.uniform(0.02, base_params['max_weight'] * 0.8) # Чуть меньше макс веса
            stock_weights_raw[ticker] = w_stock
            current_stock_weight += w_stock
        # Масштабируем акции
        if current_stock_weight > 0:
            scale_factor_stocks = stock_weight_target / current_stock_weight
            for ticker, w_stock in stock_weights_raw.items():
                scaled_w = w_stock * scale_factor_stocks
                mock_weights[ticker] = min(scaled_w, base_params['max_weight']) # Ограничиваем макс вес
                total_weight += mock_weights[ticker]

        # Финальная нормализация
        final_total = sum(mock_weights.values())
        if final_total > 0:
            mock_weights = {k: round(v / final_total, 4) for k, v in mock_weights.items() if v / final_total > 0.001}

        # Метрики
        mock_metrics = {
            'expected_return': round(random.uniform(0.05, 0.15) if strategy_profile == 'conservative' else random.uniform(0.10, 0.25), 4),
            'volatility': round(random.uniform(0.08, 0.18) if strategy_profile == 'conservative' else random.uniform(0.15, 0.30), 4),
            'sharpe_ratio': round(random.uniform(0.4, 0.9) if strategy_profile == 'conservative' else random.uniform(0.6, 1.5), 2)
        }
        
        # Создаем мок-отчет
        mock_report = {
            "weights": mock_weights,
            "metrics": mock_metrics,
            "images": {},  # Пустой словарь для изображений
            "text_report": format_portfolio_summary(mock_weights, mock_metrics),
            "portfolio_type": strategy_profile,
            "recommendations": generate_recommendations(strategy_profile, mock_metrics, mock_weights),
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"[MOCK] Simulation finished for user {user_id}. Weights: {len(mock_weights)} assets.")
        # Имитация проверки изменений
        significant_changes = True if is_initial else random.choice([True, False, False])
        
        # Сохранение мок-портфеля в БД
        _save_portfolio_results(user_id, mock_weights, mock_metrics, strategy_profile, "MOCK_Portfolio")
        
        return mock_weights, mock_metrics, significant_changes, mock_report


    # --- 3. РЕАЛЬНЫЙ ЗАПУСК PipelineExecutor ---
    try:
        # --- 3.1 Получение данных KBD ---
        kbd_data = get_cached_kbd_data() # Используем кэш
        if kbd_data.empty:
            logger.error("KBD data not available for bond processing. Cannot generate portfolio.")
            return None, None, False, None

        # --- 3.2 Получение списка тикеров ---
        tickers_final_list = get_tickers_for_user(preferences, max_stocks=preferences.max_stocks)
        if not tickers_final_list:
            logger.error(f"No tickers selected for user {user_id} based on preferences.")
            return None, None, False, None
        logger.info(f"Using tickers for user {user_id}: {tickers_final_list}")

        # --- 3.3 Инициализация PipelineExecutor ---
        # Адаптация основных параметров под профиль
        portfolio_base_params = PORTFOLIO_PARAMS.get(strategy_profile, PORTFOLIO_PARAMS['moderate'])
        executor_init_params = {
            'base_path': str(PROJECT_ROOT),
            'name': f"{strategy_profile}_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'strategy_profile': strategy_profile,
            'min_position_weight': portfolio_base_params['min_position_weight'],
            'min_rf_allocation': portfolio_base_params['min_rf_allocation'],
            'max_rf_allocation': portfolio_base_params['max_rf_allocation'],
            'risk_free_rate': COMMON_PARAMS['risk_free_rate'],
            'max_weight': portfolio_base_params['max_weight'],
            #min_assets': 5,
            #'max_assets': preferences.max_stocks
            'max_assets': portfolio_base_params['max_assets'],
            'min_assets': portfolio_base_params['min_assets'],
        }
        logger.debug(f"Initializing PipelineExecutor with params: {executor_init_params}")
        executor = PipelineExecutor(**executor_init_params)

        # --- 3.4 Обработка облигаций ---
        bond_processing_params = {
            'start_date': COMMON_PARAMS['start_date'],
            'end_date': COMMON_PARAMS['end_date'],
            'min_bonds': KBD_PARAMS['min_bonds'],
            'max_threshold': KBD_PARAMS['max_threshold'],
            'strategy_profile': strategy_profile,
            'kbd_yield_adjustment': KBD_PARAMS['kbd_yield_adjustment'],
            'update_kbd_data': False,
            'excluded_issuers': ['ВТБ', 'Мечел'],
            'n_bonds': preferences.max_bonds,
            'kbd_data': kbd_data,
            'portfolio_stability': KBD_PARAMS['portfolio_stability'],
            'use_kbd_recommendations': True,
            'kbd_duration_flexibility': KBD_PARAMS['kbd_duration_flexibility'],
            'max_adjustment_iterations': KBD_PARAMS['max_adjustment_iterations'],
            'weighting_strategy': None,
            'override_params': None,
            'output_format': 'all' 
        }
        logger.info(f"Processing bond pipeline for user {user_id}...")
        bond_results_for_run = None
        try:
            bond_results = executor.process_bond_pipeline(**bond_processing_params)
            if bond_results is not None:
                logger.info(f"Bond pipeline successful.")
                bond_results_for_run = bond_results
            else:
                logger.warning(f"Bond pipeline returned no results for user {user_id}. Proceeding without bonds.")
        except Exception as bond_exc:
            logger.exception(f"Error executing bond pipeline for user {user_id}: {bond_exc}")
            logger.warning("Proceeding without bonds due to error.")

        # --- 3.5 Основной пайплайн (генерация, оптимизация) ---
        logger.info(f"Running main portfolio pipeline for user {user_id}...")

        # Формируем параметры для каждого типа портфеля
        standard_portfolio_params = {
            'risk_free_rate': COMMON_PARAMS['risk_free_rate'],
            'min_rf_allocation': portfolio_base_params['min_rf_allocation'],
            'max_rf_allocation': portfolio_base_params['max_rf_allocation'], 
            'max_weight': portfolio_base_params['max_weight'],
            'include_short_selling': False
        }
        
        short_portfolio_params = {
            'risk_free_rate': COMMON_PARAMS['risk_free_rate'],
            'train_period': COMMON_PARAMS['train_period'],
            'test_period': COMMON_PARAMS['test_period'],
            'best_params_file': None,
            'verify_with_honest_backtest': False # Для скорости
        }
        
        combined_portfolio_params = {
            'risk_free_rate': COMMON_PARAMS['risk_free_rate'],
            'min_rf_allocation': portfolio_base_params['min_rf_allocation'],
            'max_rf_allocation': portfolio_base_params['max_rf_allocation'],
            'max_weight': portfolio_base_params['max_weight'],
            'long_ratio': 0.7,
            'include_short_selling': True
        }

        try:
            pipeline_results = executor.run_pipeline(
                tickers_list=tickers_final_list,
                bond_results=bond_results_for_run,
                strategy_profile=strategy_profile,
                signal_params=SIGNAL_PARAMS,
                standard_portfolio_params=standard_portfolio_params,
                short_portfolio_params=short_portfolio_params,
                combined_portfolio_params=combined_portfolio_params,
                optimization_params=OPTIMIZATION_PARAMS,
                # Отключение честного бэктеста для скорости
                portfolio_controls={
                    'run_standard_portfolio': True,
                    'run_short_portfolio': preferences.allow_short,
                    'run_combined_portfolio': preferences.allow_short and strategy_profile in ['moderate', 'aggressive'],
                    'override_risk_profile': False
                },
                select_portfolio_params=SELECT_PARAMS_BY_PROFILE[strategy_profile], # Выбор по профилю
                report_params={
                    'include_charts': True,
                    'include_metrics': True,
                    'include_weights': True,
                    'report_format': 'md'
                },
                visualization_params={
                    'plot_style': 'ggplot',
                    'chart_size': (12, 8),
                    'dpi': 300
                },
                # min_assets=5,
                # max_assets=preferences.max_stocks
                min_assets=executor_init_params['min_assets'],
                max_assets=executor_init_params['max_assets']
            )
        except Exception as main_exc:
            logger.exception(f"Error executing main pipeline for user {user_id}: {main_exc}")
            return None, None, False, None

        # --- 4. Обработка и Сохранение Результатов ---
        if not pipeline_results or 'best_portfolio' not in pipeline_results:
            logger.error(f"Main pipeline returned empty or invalid results for user {user_id}. Result: {pipeline_results}")
            return None, None, False, None

        best_portfolio_data = pipeline_results['best_portfolio']

        if not isinstance(best_portfolio_data, dict) or 'type' not in best_portfolio_data:
            logger.error(f"Invalid best_portfolio structure for user {user_id}: {best_portfolio_data}")
            return None, None, False, None

        portfolio_type = best_portfolio_data.get('type')
        portfolio_data = best_portfolio_data.get('portfolio', {})
        metrics_data = best_portfolio_data.get('metrics', {})

        if not isinstance(portfolio_data, dict) or 'weights' not in portfolio_data:
            logger.error(f"Missing weights in portfolio data for user {user_id}")
            return None, None, False, None

        final_weights = portfolio_data['weights']
        final_metrics = metrics_data

        if not isinstance(final_weights, dict):
            logger.error(f"Invalid weights format in {portfolio_type} portfolio for user {user_id}")
            return None, None, False, None

        # --- 4.1 Проверка значительных изменений ---
        significant_changes = False
        db_check = SessionLocal()
        try:
            latest_portfolio_db = crud.get_latest_portfolio(db_check, user_id)
            if not is_initial and latest_portfolio_db:
                significant_changes = check_significant_portfolio_changes(latest_portfolio_db.weights, final_weights)
            elif is_initial:
                significant_changes = True
            logger.info(f"Significant changes check for user {user_id}: {significant_changes}")
        except Exception as check_err:
            logger.error(f"Error checking significant changes for user {user_id}: {check_err}")
        finally:
            db_check.close()

        # --- 4.2 Сохранение в БД ---
        success_save = _save_portfolio_results(
            user_id=user_id,
            weights=final_weights,
            metrics=final_metrics,
            strategy_profile=strategy_profile,
            pipeline_name=executor_init_params['name']
        )
        if not success_save:
            logger.error(f"Failed to save portfolio results to DB for user {user_id}. Returning result anyway.")

        # --- 4.3 Генерация полного отчета ---
        # Получаем путь к директории с результатами пайплайна
        run_path = Path(pipeline_results.get('run_dir', ''))
        if not run_path.exists():
            logger.warning(f"Results directory not found: {run_path}. Will create basic report.")
            # Создаем базовый отчет без визуализаций
            report = {
                "weights": final_weights,
                "metrics": final_metrics,
                "images": {},
                "text_report": format_portfolio_summary(final_weights, final_metrics),
                "portfolio_type": portfolio_type,
                "recommendations": generate_recommendations(portfolio_type, final_metrics, final_weights),
                "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            report = generate_full_report(run_path)

        logger.info(f"Generated portfolio for user {user_id}: weights={final_weights}, metrics={final_metrics}")
        if not final_weights or len(final_weights) == 0:
            logger.error(f"Empty portfolio generated for user {user_id}!")

        return final_weights, final_metrics, significant_changes, report

    except Exception as e:
        logger.exception(f"Unhandled exception during portfolio update for user {user_id}: {e}")
        return None, None, False, None


# --- Вспомогательные Функции ---

def check_significant_portfolio_changes(old_weights: Optional[dict], new_weights: dict, threshold=0.05) -> bool:
    """Сравнивает два словаря весов."""
    if old_weights is None: return True
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    if not all_tickers: return False
    for ticker in all_tickers:
        diff = abs(old_weights.get(ticker, 0) - new_weights.get(ticker, 0))
        if diff >= threshold:
            logger.debug(f"Significant change detected for {ticker}: {diff:.2%}")
            return True
    return False


def get_tickers_for_user(preferences: Optional[UserPreferences], max_stocks: int = 10) -> list[str]:
    """Возвращает список тикеров с учетом предпочтений пользователя."""
    base_tickers = TICKERS.copy()
    
    if preferences:
        if preferences.excluded_sectors and isinstance(preferences.excluded_sectors, list):
            logger.warning("Sector-based filtering not implemented yet")
            
        if preferences.preferred_sectors and isinstance(preferences.preferred_sectors, list):
            logger.warning("Sector-based preference not implemented yet")
            
        if preferences.excluded_tickers and isinstance(preferences.excluded_tickers, list):
            base_tickers = [t for t in base_tickers if t not in preferences.excluded_tickers]
            
        if preferences.preferred_tickers and isinstance(preferences.preferred_tickers, list):
            preferred = [t for t in preferences.preferred_tickers if t not in preferences.excluded_tickers]
            base_tickers = [t for t in base_tickers if t not in preferred]
            base_tickers = preferred + base_tickers
        
        max_stocks_pref = preferences.max_stocks if preferences.max_stocks else max_stocks
        if len(base_tickers) > max_stocks_pref:
            base_tickers = base_tickers[:max_stocks_pref]
    
    # Обрезаем до max_stocks (если preferences.max_stocks не задан)
    elif len(base_tickers) > max_stocks:
        base_tickers = base_tickers[:max_stocks]
    
    return base_tickers


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running pipeline_runner.py directly for global data update test...")
    success = run_global_data_update()
    if success:
        logger.info("Global data update test finished SUCCESSFULLY.")
    else:
        logger.error("Global data update test FAILED.")
