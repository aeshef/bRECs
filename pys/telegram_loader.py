import os
import pandas as pd
import glob
from typing import Dict, List
import datetime

def load_telegram_results(
    data_dir: str = "/Users/aeshef/Documents/GitHub/kursach/data/telegram_news",
    tickers: List[str] = None,
    channel: str = "cbrstocks",
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    latest_only: bool = True
) -> Dict[str, pd.DataFrame]:
    
    """
    Загружает результаты парсинга Telegram из CSV файлов
    
    Args:
        data_dir: Директория с данными
        tickers: Список тикеров (если None, загружает все доступные)
        latest_only: Загружать только самые последние файлы для каждого тикера
        
    Returns:
        Словарь {тикер: DataFrame с сообщениями}
    """
    if not os.path.exists(data_dir):
        print(f"Директория {data_dir} не существует")
        return {}
    
    available_tickers = set()
    ticker_files = glob.glob(os.path.join(data_dir, "*_telegram_news_*.csv"))
    
    for file_path in ticker_files:
        file_name = os.path.basename(file_path)
        ticker = file_name.split('_')[0]
        available_tickers.add(ticker)
    
    if not available_tickers:
        print(f"Нет файлов с результатами в {data_dir}")
        return {}
    
    if tickers is None:
        tickers = list(available_tickers)
    
    print(f"Загрузка данных для тикеров: {', '.join(tickers)}")
    
    results = {}
    
    for ticker in tickers:
        if ticker not in available_tickers:
            print(f"Данные для тикера {ticker} не найдены, пропускаем")
            continue
        
        ticker_pattern = os.path.join(data_dir, f"{ticker}_telegram_news_*.csv")
        files = glob.glob(ticker_pattern)
        
        if not files:
            print(f"Файлы для тикера {ticker} не найдены, пропускаем")
            continue
        
        files.sort()
        
        if latest_only:
            files = [files[-1]]
        
        dfs = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                
                for col in ['tickers', 'tickers_from_tags', 'tickers_from_keywords']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) and pd.notna(x) else x)
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                dfs.append(df)
                print(f"Загружено {len(df)} строк из {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {str(e)}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            if 'date' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date
                if start_date:
                    combined_df = combined_df[combined_df['date'] >= start_date]
                if end_date:
                    combined_df = combined_df[combined_df['date'] <= end_date]

            if 'id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset='id')
            if 'date' in combined_df.columns:
                combined_df = combined_df.sort_values('date', ascending=False)
            
            results[ticker] = combined_df
    
    return results
