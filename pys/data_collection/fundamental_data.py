import os
import re
import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class SmartLabYearlyParser(BaseLogger):
    def __init__(self, ticker, base_path=f'/{BASE_PATH}/data/processed_data', needed_prc_per_year=0.9):
        super().__init__('SmartLabYearlyParser')
        self.ticker = ticker.upper()
        self.url = f"https://smart-lab.ru/q/{self.ticker}/f/y/"
        self.base_path = base_path
        # self.logger = self._setup_logger()
        self.needed_prc_per_year = needed_prc_per_year

    def fetch_page(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        try:
            self.logger.debug(f"Sending request to {self.url}")
            response = requests.get(self.url, headers=headers)
            self.logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Проверяем содержимое страницы
                self.logger.debug(f"Page title: {soup.title.string if soup.title else 'No title'}")
                all_tables = soup.find_all("table")
                self.logger.debug(f"Found {len(all_tables)} table elements in HTML")
                return soup
            self.logger.error(f"Failed to fetch data: HTTP {response.status_code}")
            return None
        except Exception as e:
            self.logger.error(f"Error during request: {e}")
            return None


    def parse_yearly_tables(self, soup):
        if soup is None:
            return pd.DataFrame()

        tables = soup.find_all("table")
        self.logger.debug(f"Found {len(tables)} tables in HTML")
        
        if not tables:
            self.logger.warning("No tables found on the page")
            divs = soup.find_all("div", class_="table")
            self.logger.debug(f"Found {len(divs)} div elements with class 'table'")
            return pd.DataFrame()

        dataframes = []
        for i, table in enumerate(tables):
            try:
                self.logger.debug(f"Attempting to parse table #{i+1}")
                self.logger.debug(f"Table HTML snippet: {str(table)[:100]}...")
                df_list = pd.read_html(str(table), header=None, decimal=',')
                if df_list:
                    self.logger.debug(f"Successfully parsed table #{i+1} into DataFrame with shape {df_list[0].shape}")
                    dataframes.append(df_list[0])
                else:
                    self.logger.debug(f"pd.read_html returned empty list for table #{i+1}")
            except Exception as e:
                self.logger.debug(f"Failed to parse table #{i+1}: {e}")
                continue

        if dataframes:
            return pd.concat(dataframes, ignore_index=True, sort=False)
        return pd.DataFrame()


    @staticmethod
    def to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(" ", "").replace("%", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return np.nan

    def clean_table(self, df):
        if df.empty:
            return df

        df_clean = df.copy()
        header_index = None
        
        for i in range(min(5, len(df_clean))):
            row = df_clean.iloc[i].astype(str).tolist()
            years_found = [cell for cell in row if re.search(r'\b\d{4}\b', cell)]
            if len(years_found) >= 2:
                header_index = i
                break

        if header_index is not None:
            new_header = df_clean.iloc[header_index].tolist()
            df_clean = df_clean.iloc[header_index+1:].reset_index(drop=True)
            df_clean.columns = new_header
        else:
            new_header = df_clean.iloc[0].tolist()
            df_clean = df_clean.iloc[1:].reset_index(drop=True)
            df_clean.columns = new_header

        seen = {}
        new_columns = []
        for item in df_clean.columns:
            if item in seen:
                seen[item] += 1
                new_columns.append(f"{item}_{seen[item]}")
            else:
                seen[item] = 0
                new_columns.append(str(item))
    
        df_clean.columns = new_columns
        return df_clean

    def preprocess_dataframe(self, df):
        if df.empty:
            return {}

        df_clean = self.clean_table(df)
        processed_data = {}
        
        indicator_col = df_clean.columns[0]
        needed_years = ['2023', '2024', '2025', '2026']

        for col in df_clean.columns[1:]:
            col_str = str(col).strip()
            match = re.search(r'\b(\d{4})\b', col_str)
            if match:
                year = match.group(1)
                if year in needed_years:  
                    year_df = df_clean[[indicator_col, col]].copy()
                    year_df.columns = ["Показатель", "Значение"]
                    year_df["Значение"] = year_df["Значение"].replace({"?": np.nan, "": np.nan})
                    year_df["value_float"] = year_df["Значение"].apply(self.to_float)
                    processed_data[year] = year_df
        
        return processed_data


class FundamentalPipeline(BaseLogger):
    def __init__(self, base_path=f'{BASE_PATH}/data/processed_data', needed_prc_per_year=0.9):
        """Initialize the pipeline with base path and percentage threshold"""
        super().__init__('FundamentalPipeline')
        self.base_path = base_path
        self.needed_prc_per_year = needed_prc_per_year
        # self.logger = self._setup_logger()
        self.ticker_data_dict = {}
        self.parser_objects = {}
        
    def process_tickers(self, ticker_list):
        """Process a list of tickers by fetching and parsing their data"""
        self.logger.info(f"Processing tickers: {ticker_list}")
        
        for ticker in ticker_list:
            parser = SmartLabYearlyParser(ticker, self.base_path, self.needed_prc_per_year)
            self.parser_objects[ticker] = parser
            
            self.logger.info(f"Fetching data for {ticker}")
            soup = parser.fetch_page()
            if soup is None:
                self.ticker_data_dict[ticker] = {}
                self.logger.warning(f"No data fetched for {ticker}")
                continue
            
            df = parser.parse_yearly_tables(soup)
            if df.empty:
                self.ticker_data_dict[ticker] = {}
                self.logger.warning(f"No tables found for {ticker}")
                continue
            
            processed_data = parser.preprocess_dataframe(df)
            self.ticker_data_dict[ticker] = processed_data
            self.logger.info(f"Successfully processed {ticker}")
            
        self._log_indicators_by_ticker()
        return self.ticker_data_dict
        
    def _log_indicators_by_ticker(self):
        """Log available indicators for each ticker and year"""
        self.logger.debug("Available indicators (with non-NaN value_float) for each ticker and year:")
        for ticker, data_for_ticker in self.ticker_data_dict.items():
            if not data_for_ticker:
                self.logger.debug(f"  {ticker} -> No data available.")
                continue
            
            for year, df_year in data_for_ticker.items():
                inds = df_year[df_year["value_float"].notna()]["Показатель"].unique()
                self.logger.debug(f"  {ticker}, {year} -> {len(inds)} indicators: {list(inds)}")
                
    def analyze_common_indicators(self):
        """Analyze and identify common indicators across tickers by year"""
        all_years = set()
        for ticker_data in self.ticker_data_dict.values():
            all_years.update(ticker_data.keys())
            
        indicators_frequency_by_year = {}
        common_indicators_by_year = {}
        
        for year in sorted(all_years):
            indicators_counter = Counter()
            
            for ticker, ticker_data in self.ticker_data_dict.items():
                data_for_year = ticker_data.get(year, None)
                if data_for_year is not None:
                    available = set(data_for_year[data_for_year["value_float"].notna()]["Показатель"].unique())
                    for indicator in available:
                        indicators_counter[indicator] += 1
            
            indicators_frequency_by_year[year] = indicators_counter
            
            non_empty_tickers = sum(1 for ticker_data in self.ticker_data_dict.values() 
                                   if year in ticker_data)
            
            if non_empty_tickers == len(self.ticker_data_dict):
                min_count = non_empty_tickers
            else:
                min_count = max(int(non_empty_tickers * self.needed_prc_per_year), 1)
            
            common_inds = [ind for ind, count in indicators_counter.items() 
                          if count >= min_count]
            
            common_indicators_by_year[year] = common_inds
            
        self._log_yearly_results(all_years, common_indicators_by_year, indicators_frequency_by_year)
        return all_years, common_indicators_by_year, indicators_frequency_by_year
        
    def _log_yearly_results(self, all_years, common_indicators_by_year, indicators_frequency_by_year):
        """Log results of common indicators analysis by year"""
        self.logger.info("Results by year:")
        for year in sorted(all_years):
            non_empty_tickers = sum(1 for ticker_data in self.ticker_data_dict.values() 
                                   if year in ticker_data)
            
            common_inds = common_indicators_by_year[year]
            
            self.logger.info(f"  Year {year}:")
            self.logger.info(f"    - Data available for {non_empty_tickers} out of {len(self.ticker_data_dict)} tickers")
            self.logger.info(f"    - {len(common_inds)} common indicators (for most tickers)")
            
            for ind in sorted(common_inds):
                freq = indicators_frequency_by_year[year][ind]
                self.logger.info(f"      * {ind} (found in {freq} tickers)")

    def visualize_common_indicators(self, all_years, common_indicators_by_year):
        """Create visualizations for common indicators across tickers by year"""
        viz_path = os.path.join(os.path.dirname(self.base_path), 'fundamental_viz')
        for year in all_years:
            year_viz_path = os.path.join(viz_path, year)
            os.makedirs(year_viz_path, exist_ok=True)

            common_indicators = common_indicators_by_year[year]
            data = []

            for ticker in self.ticker_data_dict:
                if year in self.ticker_data_dict[ticker]:
                    df_year = self.ticker_data_dict[ticker][year]
                    df_filtered = df_year[df_year['Показатель'].isin(common_indicators)]
                    df_filtered['Ticker'] = ticker
                    data.append(df_filtered)

            if data:
                full_df = pd.concat(data)
                full_df['value_float'] = pd.to_numeric(full_df['value_float'], errors='coerce')
                full_df = full_df.dropna(subset=['value_float'])

                plots = [
                    ('P/E', 'ROE, %'),
                    ('P/BV', 'ROA, %'),
                    ('Див доход, ао, %', 'P/E'),
                    ('Капитализация, млрд руб', 'Чистая прибыль, млрд руб'),
                    ('EV/EBITDA', 'Рентаб EBITDA, %')
                ]

                for x_axis, y_axis in plots:
                    if x_axis in common_indicators and y_axis in common_indicators:
                        plt.figure(figsize=(12, 8))
                        plot_df = full_df.pivot_table(index='Ticker', columns='Показатель', values='value_float').reset_index()
                        sns.scatterplot(data=plot_df, x=x_axis, y=y_axis, hue='Ticker', s=200, palette='tab20', legend='full')
                        plt.title(f'{x_axis} vs {y_axis} for {year}')
                        plt.xlabel(x_axis)
                        plt.ylabel(y_axis)
                        plt.grid(True, alpha=0.3)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        for line in range(plot_df.shape[0]):
                            plt.text(plot_df[x_axis][line], plot_df[y_axis][line], plot_df['Ticker'][line], fontsize=8)
                        plt.savefig(os.path.join(year_viz_path, f'{x_axis.replace("/", "_").replace("%", "")}_vs_{y_axis.replace("/", "_").replace("%", "")}.png'), bbox_inches='tight', dpi=300)
                        plt.close()

                for metric in common_indicators:
                    plt.figure(figsize=(14, 8))
                    plot_df = full_df[full_df['Показатель'] == metric].sort_values('value_float')
                    if not plot_df.empty:
                        sns.barplot(data=plot_df, x='value_float', y='Ticker', palette='viridis')
                        plt.title(f'{metric} - {year}')
                        plt.xlabel('Значение')
                        plt.ylabel('Компании')
                        plt.grid(axis='x', alpha=0.3)
                        for p in plt.gca().patches:
                            width = p.get_width()
                            plt.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.1f}', va='center')
                        plt.savefig(os.path.join(year_viz_path, f'{metric.replace("/", "_").replace("%", "")}.png'), bbox_inches='tight', dpi=300)
                        plt.close()

        self.logger.info(f"Визуализации сохранены в: {viz_path}")
                
    def save_data(self, all_years, common_indicators_by_year):
        """Save processed data to files for each ticker and year"""
        for ticker, parser in self.parser_objects.items():
            ticker_year_data = self.ticker_data_dict.get(ticker, {})

            for year in sorted(all_years):
                year_folder_path = os.path.join(self.base_path, ticker, "fundamental_analysis", year)
                os.makedirs(year_folder_path, exist_ok=True)
                
                all_file_path = os.path.join(year_folder_path, "all.csv")
                common_file_path = os.path.join(year_folder_path, "common.csv")
                
                if year in ticker_year_data:
                    df_year = ticker_year_data[year].copy()
                    all_df = df_year[df_year["value_float"].notna()][["Показатель", "value_float"]]
                    all_df.to_csv(all_file_path, index=False, encoding="utf-8")
                else:
                    pd.DataFrame(columns=["Показатель", "value_float"]).to_csv(
                        all_file_path, index=False, encoding="utf-8"
                    )
                
                common_inds = common_indicators_by_year[year]
                
                if year in ticker_year_data:
                    df_year = ticker_year_data[year].copy()
                    
                    result_rows = []
                    
                    for indicator in sorted(common_inds):
                        matching_rows = df_year[df_year["Показатель"] == indicator]
                        
                        if len(matching_rows) > 0 and not pd.isna(matching_rows["value_float"].iloc[0]):
                            result_rows.append({
                                "Показатель": indicator,
                                "value_float": matching_rows["value_float"].iloc[0]
                            })
                        else:
                            result_rows.append({
                                "Показатель": indicator,
                                "value_float": "NO_DATA"
                            })
                    
                    result_df = pd.DataFrame(result_rows)
                    result_df.to_csv(common_file_path, index=False, encoding="utf-8")
                    
                else:
                    placeholder_df = pd.DataFrame({
                        "Показатель": common_inds,
                        "value_float": ["NO_DATA"] * len(common_inds)
                    })
                    placeholder_df.to_csv(common_file_path, index=False, encoding="utf-8")
                    
        self.logger.info("Process completed. Created all.csv and common.csv files for each year and ticker.")
        self.logger.info("Used 'NO_DATA' token for missing indicators.")


def run_pipeline_fundamental(ticker_list, base_path=f'{BASE_PATH}/data/processed_data'):
    pipeline = FundamentalPipeline(base_path)
    pipeline.process_tickers(ticker_list)
    all_years, common_indicators_by_year, _ = pipeline.analyze_common_indicators()
    pipeline.save_data(all_years, common_indicators_by_year)
    pipeline.visualize_common_indicators(all_years, common_indicators_by_year)
