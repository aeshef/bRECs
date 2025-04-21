import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List
import importlib

import technical_indicators
importlib.reload(technical_indicators)
from technical_indicators import TechnicalIndicators

class TechAnalysisPipeline:
    def __init__(self,
                 base_dir: str = "/Users/aeshef/Documents/GitHub/kursach",
                 tickers: List[str] = [
                     "GAZP", "SBER", "LKOH", "GMKN", "ROSN", "TATN", "MTSS",
                     "ALRS", "SNGS", "VTBR", "NVTK", "POLY", "MVID", "PHOR",
                     "SIBN", "AFKS", "MAGN", "RUAL"
                 ]):
        self.base_dir = base_dir
        self.tickers = tickers
        self.results = {}

    def run_pipeline(self):
        print("=== НАЧАЛО ТЕХНИЧЕСКОГО АНАЛИЗА ===")
        for ticker in self.tickers:
            print(f"\nОбработка тикера: {ticker}")
            ticker_dir = os.path.join(self.base_dir, "data", "processed_data", ticker)
            if not os.path.exists(ticker_dir):
                print(f"Директория для тикера {ticker} не найдена. Пропускаем!")
                continue

            parquet_files = [f for f in os.listdir(ticker_dir)
                             if f.endswith(".parquet") and ticker in f]
            if not parquet_files:
                print(f"Нет файлов .parquet для тикера {ticker}")
                continue

            parquet_file = sorted(parquet_files,
                                  key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)),
                                  reverse=True)[0]
            file_path = os.path.join(ticker_dir, parquet_file)
            print(f"Используем файл: {parquet_file}")

            tech_out_dir = os.path.join(ticker_dir, "tech_analysis")
            os.makedirs(tech_out_dir, exist_ok=True)

            try:
                tech_pipeline = TechnicalIndicators(file_path)
                tech_pipeline.load_data()
                tech_pipeline.calculate_indicators()

                indicators_csv_path = os.path.join(tech_out_dir, f"{ticker}_tech_indicators.csv")
                tech_pipeline.df.to_csv(indicators_csv_path)
                print(f"Сохранен CSV с индикаторами: {indicators_csv_path}")

                plt.figure(figsize=(14, 7))
                plt.plot(tech_pipeline.df.index, tech_pipeline.df["close"], label="Close", color="blue")
                plt.plot(tech_pipeline.df.index, tech_pipeline.df["SMA_20"], label=f"SMA_20 ({tech_pipeline.sma_window})", color="orange")
                plt.plot(tech_pipeline.df.index, tech_pipeline.df["EMA_20"], label=f"EMA_20 ({tech_pipeline.sma_window})", color="green")
                plt.fill_between(tech_pipeline.df.index,
                                 tech_pipeline.df["BB_lower"],
                                 tech_pipeline.df["BB_upper"],
                                 color="grey", alpha=0.3, label="Bollinger Bands")
                plt.xlabel("Дата")
                plt.ylabel("Цена")
                plt.title(f"{ticker}: Цена с техническими индикаторами")
                plt.legend()
                plt.grid(True)
                price_graph_path = os.path.join(tech_out_dir, f"{ticker}_price_indicators.png")
                plt.savefig(price_graph_path)
                plt.close()
                print(f"Сохранен график цены с индикаторами: {price_graph_path}")

                plt.figure(figsize=(14, 4))
                plt.plot(tech_pipeline.df.index, tech_pipeline.df["MACD"], label="MACD", color="blue")
                plt.plot(tech_pipeline.df.index, tech_pipeline.df["MACD_signal"], label="MACD Signal", color="red")
                plt.bar(tech_pipeline.df.index, tech_pipeline.df["MACD_diff"],
                        label="MACD Diff", color="grey", alpha=0.5)
                plt.xlabel("Дата")
                plt.ylabel("MACD Value")
                plt.title(f"{ticker}: MACD")
                plt.legend()
                plt.grid(True)
                macd_graph_path = os.path.join(tech_out_dir, f"{ticker}_MACD.png")
                plt.savefig(macd_graph_path)
                plt.close()
                print(f"Сохранен график MACD: {macd_graph_path}")

                self.results[ticker] = {
                    "indicators_csv": indicators_csv_path,
                    "price_graph": price_graph_path,
                    "macd_graph": macd_graph_path,
                }
            except Exception as e:
                print(f"Ошибка при обработке тикера {ticker}: {e}")

        self.create_summary_report()
        print("=== ТЕХНИЧЕСКИЙ АНАЛИЗ ЗАВЕРШЕН ===")

    def create_summary_report(self):
        summary_lines = []
        summary_lines.append("=== Сводный отчет по техническому анализу ===")
        summary_lines.append(f"Дата анализа: {datetime.now()}\n")
        for ticker, paths in self.results.items():
            summary_lines.append(f"Тикер: {ticker}")
            summary_lines.append(f"CSV с индикаторами: {paths.get('indicators_csv', 'Не найден')}")
            summary_lines.append(f"График цены с индикаторами: {paths.get('price_graph', 'Не найден')}")
            summary_lines.append(f"График MACD: {paths.get('macd_graph', 'Не найден')}")
            summary_lines.append("-" * 50)
        report_path = os.path.join(self.base_dir, "data", "processed_data", "tech_analysis_summary.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(summary_lines))
        print(f"Сводный отчет сохранен в {report_path}")
