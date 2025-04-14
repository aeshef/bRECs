# telegram_parser.py

import os
import re
import json
import pandas as pd
import logging
import datetime
import time
import sys
import argparse
from typing import List, Dict, Any

from telethon.sync import TelegramClient
from telethon import functions, types

def collect_telegram_news(
    api_id: int,
    api_hash: str,
    channel: str = "cbrstocks",
    limit: int = 50000,
    output_dir: str = "/Users/aeshef/Documents/GitHub/kursach/data/telegram_news",
    tickers: List[str] = None,
    start_date: datetime.date = None,  # Новый параметр
    end_date: datetime.date = None     # Новый параметр
) -> Dict[str, pd.DataFrame]:
    
    """
    Сбор новостей из Telegram-канала
    
    Args:
        api_id: API ID для Telegram
        api_hash: API Hash для Telegram
        channel: Имя канала (без @)
        limit: Максимальное количество сообщений для сбора
        output_dir: Директория для сохранения результатов
        tickers: Список тикеров для фильтрации
    
    Returns:
        Dict[str, pd.DataFrame]: словарь {тикер: DataFrame с сообщениями}
    """
    # Проверка наличия необходимых пакетов
    try:
        import pandas as pd
        from telethon.sync import TelegramClient
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Пожалуйста, убедитесь, что пакеты pandas и telethon установлены:")
        print("pip install pandas telethon")
        return {}
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('Telegram_Parser')
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Словарь компаний и их ключевых слов
    COMPANY_INFO = {
        'AFKS': {'name': 'АФК "Система"', 'industry': 'конгломерат', 
                'keywords': ['афк система', 'система', 'евтушенков']},
        'ALRS': {'name': 'АЛРОСА', 'industry': 'алмазы', 
                'keywords': ['алроса', 'алмаз', 'якутия']},
        'GMKN': {'name': 'Норникель', 'industry': 'металлургия', 
                'keywords': ['норильский никель', 'норникель', 'потанин']},
        'LKOH': {'name': 'Лукойл', 'industry': 'нефть', 
                'keywords': ['лукойл', 'алекперов', 'нефтехимия']},
        'MAGN': {'name': 'ММК', 'industry': 'металлургия', 
                'keywords': ['магнитогорский металлургический комбинат', 'ммк', 'рашников']},
        'MTSS': {'name': 'МТС', 'industry': 'телекоммуникации', 
                'keywords': ['мтс', 'мобильные телесистемы', 'телеком']},
        'MVID': {'name': 'М.Видео', 'industry': 'ритейл', 
                'keywords': ['м.видео', 'мвидео', 'электроника']},
        'NVTK': {'name': 'Новатэк', 'industry': 'газ', 
                'keywords': ['новатэк', 'спг', 'михеев']},
        'PHOR': {'name': 'ФосАгро', 'industry': 'химия', 
                'keywords': ['фосагро', 'удобрения', 'гурьев']},
        'ROSN': {'name': 'Роснефть', 'industry': 'нефть', 
                'keywords': ['роснефть', 'сечин', 'нефтяная компания']},
        'RUAL': {'name': 'Русал', 'industry': 'металлургия', 
                'keywords': ['русал', 'алюминий', 'денипаска']},
        'SIBN': {'name': 'Газпром нефть', 'industry': 'нефть', 
                'keywords': ['газпром нефть', 'нефтедобыча', 'дыбенко']},
        'SNGS': {'name': 'Сургутнефтегаз', 'industry': 'нефть', 
                'keywords': ['сургутнефтегаз', 'богданов', 'нефтяные запасы']},
        'TATN': {'name': 'Татнефть', 'industry': 'нефть', 
                'keywords': ['татнефть', 'татарстан', 'минниханов']},
        'VTBR': {'name': 'ВТБ', 'industry': 'банк', 
                'keywords': ['втб', 'внешторгбанк', 'костин']}
    }

    # Дефолтные тикеры если не указаны
    if tickers is None:
        tickers = [
            'AFKS', 'ALRS', 'GMKN', 'LKOH', 'MAGN', 
            'MTSS', 'MVID', 'NVTK', 'PHOR', 'ROSN', 
            'RUAL', 'SIBN', 'SNGS', 'TATN', 'VTBR'
        ]

    
    try:
        with TelegramClient('telegram_session', api_id, api_hash) as client:
            logger.info(f"Подключено к Telegram API")
            entity = client.get_entity(channel)

            # Собираем сообщения с ручной фильтрацией по дате
            messages = []
            for message in client.iter_messages(entity, limit=limit):
                msg_date = message.date.date()
                
                # Пропускаем сообщения вне диапазона
                if start_date and msg_date < start_date:
                    continue
                if end_date and msg_date > end_date:
                    continue
                
                messages.append(message)
                time.sleep(0.1)
                
                if len(messages) % 20 == 0:
                    logger.info(f"Получено {len(messages)} сообщений...")

            logger.info(f"Всего получено {len(messages)} сообщений после фильтрации")
            
            # Преобразуем сообщения в DataFrame
            message_data = []
            
            for msg in messages:
                if not msg.text:
                    continue
                    
                # Ищем тикеры в тексте по хэштегам #TICKER
                tickers_from_tags = []
                ticker_pattern = r'#([A-Z0-9]{4,6})'
                found_tickers = re.findall(ticker_pattern, msg.text)
                
                # Фильтруем только известные тикеры
                tickers_from_tags = [t for t in found_tickers if t in COMPANY_INFO]
                
                # Ищем по ключевым словам если не нашли по тегам
                tickers_from_keywords = []
                if not tickers_from_tags:
                    text_lower = msg.text.lower()
                    for ticker, info in COMPANY_INFO.items():
                        if info['name'].lower() in text_lower:
                            tickers_from_keywords.append(ticker)
                            continue
                            
                        for keyword in info['keywords']:
                            if keyword in text_lower:
                                tickers_from_keywords.append(ticker)
                                break
                
                # Объединяем найденные тикеры
                all_tickers = list(set(tickers_from_tags + tickers_from_keywords))
                
                # Если нашли тикеры, добавляем сообщение
                message_info = {
                    'id': msg.id,
                    'date': msg.date,
                    'text': msg.text,
                    'tickers': all_tickers,
                    'tickers_from_tags': tickers_from_tags,
                    'tickers_from_keywords': tickers_from_keywords,
                    'has_media': msg.media is not None
                }
                
                message_data.append(message_info)
            
            # Создаем DataFrame
            df = pd.DataFrame(message_data)
            
            # Сохраняем все сообщения в CSV
            if not df.empty:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                all_messages_file = os.path.join(output_dir, f"telegram_cbrstocks_{timestamp}.csv")
                df.to_csv(all_messages_file, index=False, encoding='utf-8')
                logger.info(f"Сохранено {len(df)} сообщений в файл {all_messages_file}")
            
            # Словарь для хранения результатов по тикерам
            results = {}
            
            # Фильтруем сообщения по тикерам
            for ticker in tickers:
                # Проверяем, есть ли тикер в COMPANY_INFO
                if ticker not in COMPANY_INFO:
                    logger.warning(f"Тикер {ticker} не найден в списке компаний, пропускаем")
                    continue
                    
                # Фильтруем сообщения для текущего тикера
                ticker_messages = df[df['tickers'].apply(
                    lambda x: isinstance(x, list) and ticker in x
                )].copy()
                
                if ticker_messages.empty:
                    logger.warning(f"Нет сообщений для тикера {ticker}")
                    continue
                
                # Добавляем информацию о тикере и компании
                ticker_messages['ticker'] = ticker
                ticker_messages['company_name'] = COMPANY_INFO[ticker]['name']
                ticker_messages['industry'] = COMPANY_INFO[ticker]['industry']
                
                # Помечаем тип новости
                ticker_messages['news_type'] = ticker_messages.apply(
                    lambda row: 'company_specific' if ticker in row.get('tickers_from_tags', []) 
                    else 'industry', axis=1
                )
                
                results[ticker] = ticker_messages
                
                # Сохраняем в отдельный файл
                timestamp = datetime.datetime.now().strftime("%Y%m%d")
                ticker_filename = f"{ticker}_telegram_news_{timestamp}.csv"
                ticker_path = os.path.join(output_dir, ticker_filename)
                ticker_messages.to_csv(ticker_path, index=False, encoding='utf-8')
                logger.info(f"Сохранено {len(ticker_messages)} сообщений для {ticker} в файл {ticker_path}")
            
            # Выводим статистику
            logger.info("\nРезультаты:")
            for ticker, ticker_df in results.items():
                company_messages = ticker_df[ticker_df['news_type'] == 'company_specific']
                industry_messages = ticker_df[ticker_df['news_type'] == 'industry']
                
                logger.info(f"Тикер: {ticker}")
                logger.info(f"  - Сообщений о компании: {len(company_messages)}")
                logger.info(f"  - Сообщений об индустрии: {len(industry_messages)}")
                logger.info(f"  - Всего: {len(ticker_df)}")
                logger.info("=" * 40)
            
            return results
            
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Парсер Telegram-канала @cbrstocks')
    parser.add_argument('--api_id', type=int, required=True, help='API ID для Telegram')
    parser.add_argument('--api_hash', type=str, required=True, help='API Hash для Telegram')
    parser.add_argument('--limit', type=int, default=50000, help='Лимит сообщений (по умолчанию 100)')
    parser.add_argument('--tickers', type=str, nargs='+', help='Список тикеров для фильтрации')
    
    # Добавляем аргументы для дат
    parser.add_argument('--start_date', 
                        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(),
                        help='Начальная дата периода (ГГГГ-ММ-ДД)')
    parser.add_argument('--end_date', 
                        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(),
                        help='Конечная дата периода (ГГГГ-ММ-ДД)')
    
    args = parser.parse_args()
    
    collect_telegram_news(
        api_id=args.api_id,
        api_hash=args.api_hash,
        limit=args.limit,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )