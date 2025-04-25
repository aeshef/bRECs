import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
import importlib
import logging
import sys

from pys.data_collection.technical_indicators import TechnicalIndicators
  
# current_dir = os.path.dirname(os.path.abspath(__file__))
# while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
#     current_dir = os.path.dirname(current_dir)
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
# from utils.logger import BaseLogger

# sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection')
# from private_info import BASE_PATH

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class TechAnalysisPipeline(BaseLogger):
    def __init__(self,
                 base_dir: str = BASE_PATH,
                 tickers: List[str] = [
                     "GAZP", "SBER", "LKOH", "GMKN", "ROSN", "TATN", "MTSS",
                     "ALRS", "SNGS", "VTBR", "NVTK", "POLY", "MVID", "PHOR",
                     "SIBN", "AFKS", "MAGN", "RUAL"
                 ]):
        super().__init__('TechAnalysisPipeline')
        self.base_dir = base_dir
        self.tickers = tickers
        self.results = {} 

    def run_pipeline(self):
        self.logger.info("=== НАЧАЛО ТЕХНИЧЕСКОГО АНАЛИЗА ===")

        indicators_params = {
            'SMA_20': "window: sma_window",
            'EMA_20': "window: sma_window",
            'WMA': "window: 14",
            'RSI_14': "window: rsi_window",
            'Stoch_%K': "window: 14",
            'Stoch_%D': "signal period: 14",
            'BB_upper': "BB window: sma_window, std: 2",
            'BB_mid': "BB window: sma_window",
            'BB_lower': "BB window: sma_window, std: 2",
            'ATR_14': "window: 14",
            'OBV': "N/A",
            'CMF_20': "window: 20",
            'MACD': "default settings",
            'MACD_signal': "default settings",
            'MACD_diff': "default settings",
            'ADX_14': "window: 14",
            'Vortex_14+': "window: 14",
            'Vortex_14-': "window: 14",
            'CCI_20': "window: 20",
            'Williams_%R_14': "window: 14",
            'Ichimoku_Conversion': "params: (9,26)",
            'Ichimoku_Base': "params: (9,26)",
            'KAMA_10': "window: 10",
            'ROC_10': "window: 10",
            'Ultimate_Osc': "default settings"
        }
        
        for ticker in self.tickers:
            self.logger.info(f"Обработка тикера: {ticker}")
            ticker_dir = os.path.join(self.base_dir, "data", "processed_data", ticker)
            if not os.path.exists(ticker_dir):
                self.logger.warning(f"Директория для тикера {ticker} не найдена. Пропускаем!")
                continue

            parquet_files = [f for f in os.listdir(ticker_dir)
                             if f.endswith(".parquet") and ticker in f]
            if not parquet_files:
                self.logger.warning(f"Нет файлов .parquet для тикера {ticker}")
                continue

            parquet_file = sorted(parquet_files,
                                  key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)),
                                  reverse=True)[0]
            file_path = os.path.join(ticker_dir, parquet_file)
            self.logger.info(f"Используем файл: {parquet_file}")

            tech_out_dir = os.path.join(ticker_dir, "tech_analysis")
            os.makedirs(tech_out_dir, exist_ok=True)

            try:
                tech_pipeline = TechnicalIndicators(file_path)
                tech_pipeline.load_data()
                tech_pipeline.calculate_indicators()

                df = tech_pipeline.df
                base_columns = {"open", "high", "low", "close", "volume"}
                indicator_summary = {}
                for col in df.columns:
                    if col not in base_columns:
                        missing_count = int(df[col].isna().sum())
                        total = df.shape[0]
                        param_info = indicators_params.get(col, "N/A")
                        indicator_summary[col] = {
                            "missing_values": missing_count,
                            "total_rows": total,
                            "parameter_info": param_info
                        }

                indicators_csv_path = os.path.join(tech_out_dir, f"{ticker}_tech_indicators.csv")
                df.to_csv(indicators_csv_path)
                self.logger.info(f"Сохранен CSV с индикаторами: {indicators_csv_path}")

                plt.figure(figsize=(14, 7))
                plt.plot(df.index, df["close"], label="Close", color="blue")
                plt.plot(df.index, df["SMA_20"], label=f"SMA_20 ({tech_pipeline.sma_window})", color="orange")
                plt.plot(df.index, df["EMA_20"], label=f"EMA_20 ({tech_pipeline.sma_window})", color="green")
                plt.fill_between(df.index,
                                 df["BB_lower"],
                                 df["BB_upper"],
                                 color="grey", alpha=0.3, label="Bollinger Bands")
                plt.xlabel("Дата")
                plt.ylabel("Цена")
                plt.title(f"{ticker}: Цена с техническими индикаторами")
                plt.legend()
                plt.grid(True)
                price_graph_path = os.path.join(tech_out_dir, f"{ticker}_price_indicators.png")
                plt.savefig(price_graph_path)
                plt.close()
                self.logger.info(f"Сохранен график цены с индикаторами: {price_graph_path}")

                plt.figure(figsize=(14, 4))
                plt.plot(df.index, df["MACD"], label="MACD", color="blue")
                plt.plot(df.index, df["MACD_signal"], label="MACD Signal", color="red")
                plt.bar(df.index, df["MACD_diff"], label="MACD Diff", color="grey", alpha=0.5)
                plt.xlabel("Дата")
                plt.ylabel("MACD Value")
                plt.title(f"{ticker}: MACD")
                plt.legend()
                plt.grid(True)
                macd_graph_path = os.path.join(tech_out_dir, f"{ticker}_MACD.png")
                plt.savefig(macd_graph_path)
                plt.close()
                self.logger.info(f"Сохранен график MACD: {macd_graph_path}")

                self.results[ticker] = {
                    "indicators_csv": indicators_csv_path,
                    "price_graph": price_graph_path,
                    "macd_graph": macd_graph_path,
                    "indicator_summary": indicator_summary,
                    "data_rows": df.shape[0],
                    "sma_window": tech_pipeline.sma_window,
                    "rsi_window": tech_pipeline.rsi_window,
                }
            except Exception as e:
                self.logger.error(f"Ошибка при обработке тикера {ticker}: {e}")

        self.create_summary_report()
        self.logger.info("=== ТЕХНИЧЕСКИЙ АНАЛИЗ ЗАВЕРШЕН ===")

    def create_summary_report(self):
        summary_lines = []
        summary_lines.append("=== Сводный отчет по техническому анализу ===")
        summary_lines.append(f"Дата анализа: {datetime.now()}\n")

        for ticker, details in self.results.items():
            summary_lines.append(f"Тикер: {ticker}")
            summary_lines.append(f"Общее число строк данных: {details.get('data_rows', 'N/A')}")
            summary_lines.append(f"sma_window: {details.get('sma_window', 'N/A')}, rsi_window: {details.get('rsi_window', 'N/A')}")
            summary_lines.append(f"CSV с индикаторами: {details.get('indicators_csv', 'Не найден')}")
            summary_lines.append(f"График цены с индикаторами: {details.get('price_graph', 'Не найден')}")
            summary_lines.append(f"График MACD: {details.get('macd_graph', 'Не найден')}")
            summary_lines.append("\nИнформация по вычисленным индикаторам:")
            indicator_summary = details.get("indicator_summary", {})
            for indicator, stats in indicator_summary.items():
                missing = stats.get("missing_values", 0)
                total = stats.get("total_rows", 0)
                param_info = stats.get("parameter_info", "N/A")
                summary_lines.append(f"  {indicator} -> {param_info} | Пропущено: {missing} из {total} записей")
            summary_lines.append("-" * 50)

        report_path = os.path.join(self.base_dir, "data", "processed_data", "tech_analysis_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        self.logger.info(f"Сводный отчет сохранен в {report_path}")

def run_pipeline_technical():
    TechAnalysisPipeline(base_dir=BASE_PATH).run_pipeline()