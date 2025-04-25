import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import traceback
import json
import re
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# while os.path.basename(current_dir) != 'pys' and current_dir != os.path.dirname(current_dir):
#     current_dir = os.path.dirname(current_dir)
#     if current_dir == os.path.dirname(current_dir):
#         break

# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)

# from utils.logger import BaseLogger

# sys.path.append('/Users/aeshef/Documents/GitHub/kursach/pys/data_collection')
# from private_info import BASE_PATH

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class BondsProcessor(BaseLogger):
    def __init__(self, bonds_dir=f"{BASE_PATH}/data/processed_data/bonds_csv", direct_data_file=None):
        """
        Инициализация процессора облигаций
        
        Args:
            bonds_dir: директория с файлами облигаций
            direct_data_file: прямой путь к файлу с данными (если не используются файлы в директории)
        """
        self.bonds_dir = bonds_dir
        self.direct_data_file = direct_data_file
        self.bond_files = self._get_sorted_bond_files() if direct_data_file is None else []
        self.processed_bonds = None
        self.previous_recommendations = None
        super().__init__('BondsProcessor')
        self.load_previous_recommendations()
        
    def _get_sorted_bond_files(self):
        """Get sorted bond files from the directory"""
        try:
            files = [f for f in os.listdir(self.bonds_dir) if f.endswith('.csv') or f.endswith('.txt')]
            
            # Extract date from filename pattern
            def extract_date(filename):
                # Try different date patterns in filenames
                date_patterns = [
                    r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                    r'(\d{8})',              # YYYYMMDD
                    r'(\d{2}\.\d{2}\.\d{4})' # DD.MM.YYYY
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, filename)
                    if match:
                        date_str = match.group(1)
                        try:
                            if len(date_str) == 8:  # YYYYMMDD
                                return datetime.strptime(date_str, '%Y%m%d')
                            elif '.' in date_str:  # DD.MM.YYYY
                                return datetime.strptime(date_str, '%d.%m.%Y')
                            else:  # YYYY-MM-DD
                                return datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            pass
                
                # If no date found in filename, use file modification time
                file_path = os.path.join(self.bonds_dir, filename)
                mod_time = os.path.getmtime(file_path)
                return datetime.fromtimestamp(mod_time)
            
            return sorted(files, key=extract_date)
        except Exception as e:
            self.logger.error(f"Error getting list of files: {e}")
            return []
        
    def load_previous_recommendations(self):
        """Load previous bond recommendations if available"""
        try:
            results_dir = os.path.join(os.path.dirname(self.bonds_dir), 'results')
            prev_file = os.path.join(results_dir, 'latest_recommendations.json')
            if os.path.exists(prev_file):
                with open(prev_file, 'r') as f:
                    self.previous_recommendations = json.load(f)
                self.logger.info(f"Loaded previous recommendations from {self.previous_recommendations.get('date', 'unknown date')}")
        except Exception as e:
            self.logger.warning(f"Could not load previous recommendations: {e}")
            self.previous_recommendations = None
            
    def process_direct_data(self, 
                          min_yield=6.0,
                          max_yield=22.0,
                          min_duration=0.1,
                          max_duration=5.0,
                          max_expiration_date=None,
                          min_liquidity=None,
                          excluded_issuers=None,
                          excluded_bonds=None,
                          include_only=None,
                          use_latest_snapshot_only=True,
                          score_weights=None):
        """
        Обработка данных напрямую из файла датасета
        
        Args:
            min_yield: минимальная доходность (%)
            max_yield: максимальная доходность (%)
            min_duration: минимальная дюрация (лет)
            max_duration: максимальная дюрация (лет)
            max_expiration_date: максимальная дата погашения
            min_liquidity: минимальный объем торгов
            excluded_issuers: список исключаемых эмитентов
            excluded_bonds: список исключаемых кодов облигаций
            include_only: список включаемых кодов облигаций
            use_latest_snapshot_only: использовать только последний снапшот данных
            score_weights: словарь весов для расчета скора облигаций
            
        Returns:
            DataFrame с обработанными облигациями или None в случае ошибки
        """
        if not self.direct_data_file or not os.path.exists(self.direct_data_file):
            self.logger.error(f"Direct data file not found: {self.direct_data_file}")
            return None
        
        self.logger.info(f"Loading direct data from {self.direct_data_file}")
        
        try:
            # Загружаем данные
            dataset = pd.read_csv(self.direct_data_file)
            
            # Сохраняем параметры фильтрации
            self.filter_params = {
                'min_yield': min_yield,
                'max_yield': max_yield,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'max_expiration_date': max_expiration_date,
                'min_liquidity': min_liquidity,
                'excluded_issuers': excluded_issuers or [],
                'excluded_bonds': excluded_bonds or [],
                'include_only': include_only,
                'use_latest_snapshot_only': use_latest_snapshot_only
            }
            
            # Определяем идентификационную колонку
            id_col = 'secid' if 'secid' in dataset.columns else dataset.columns[0]
            
            # Стандартизируем имена колонок
            column_mapping = {
                'secid': 'security_code',
                'name': 'full_name',
                'price': 'price_pct',
                'yield': 'yield',
                'duration': 'duration_months',
                'maturity_date': 'expiration_date',
                'volume': 'trading_volume',
                'value': 'trading_value'
            }
            
            # Переименовываем колонки, которые существуют в датасете
            rename_cols = {k: v for k, v in column_mapping.items() 
                          if k in dataset.columns and v not in dataset.columns}
            if rename_cols:
                dataset = dataset.rename(columns=rename_cols)
                self.logger.info(f"Renamed columns: {rename_cols}")
                
            # Добавляем колонку duration_years, если её нет
            if 'duration_years' not in dataset.columns:
                if 'duration_months' in dataset.columns:
                    # Преобразуем месяцы в годы
                    dataset['duration_years'] = dataset['duration_months'] / 12
                    self.logger.info("Created duration_years column from duration_months")
                elif 'duration' in dataset.columns:
                    # Определяем, в чём измеряется duration на основе максимального значения
                    max_duration = dataset['duration'].max()
                    if pd.notnull(max_duration):
                        if max_duration > 100:  # Если в днях
                            dataset['duration_years'] = dataset['duration'] / 365.25
                            self.logger.info("Created duration_years column from duration (days)")
                        else:  # Если в месяцах или годах
                            if max_duration > 30:  # Вероятно, в месяцах
                                dataset['duration_years'] = dataset['duration'] / 12
                                self.logger.info("Created duration_years column from duration (months)")
                            else:  # Вероятно, уже в годах
                                dataset['duration_years'] = dataset['duration']
                                self.logger.info("Created duration_years column from duration (already in years)")
            
                        ## В методе process_direct_data добавьте следующий код для обработки даты
            # Обрабатываем колонку даты снапшота
            if 'date' in dataset.columns and 'snapshot_date' not in dataset.columns:
                dataset['snapshot_date'] = pd.to_datetime(dataset['date'], errors='coerce')
                self.logger.info("Created snapshot_date from date column")
            elif 'snapshot_date' in dataset.columns and not pd.api.types.is_datetime64_any_dtype(dataset['snapshot_date']):
                dataset['snapshot_date'] = pd.to_datetime(dataset['snapshot_date'], errors='coerce')
                self.logger.info("Converted snapshot_date to datetime")

            
            # Используем только последний снапшот, если нужно
            if use_latest_snapshot_only and 'snapshot_date' in dataset.columns:
                latest_date = dataset['snapshot_date'].max()
                dataset = dataset[dataset['snapshot_date'] == latest_date]
                self.logger.info(f"Filtered data to use only latest snapshot: {latest_date}")
                
            # Применяем фильтры
            filtered_data = self._apply_custom_filters(dataset)
            
            # Рассчитываем метрики
            self.processed_bonds = self.calculate_bond_metrics(filtered_data, score_weights)
            self.logger.info(f"Processed {len(self.processed_bonds)} bonds from direct data")
            
            # Определяем рыночные условия
            self.market_condition = self._detect_market_condition()
            self.logger.info(f"Detected market condition: {self.market_condition}")
            
            return self.processed_bonds
            
        except Exception as e:
            self.logger.error(f"Error processing direct data: {e}\n{traceback.format_exc()}")
            return None
    
    def _apply_custom_filters(self, df):
        """
        Применить фильтры к данным
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с отфильтрованными данными
        """
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        self.logger.info(f"Starting filters with {initial_count} bonds")
        
        # Фильтр по доходности
        if self.filter_params['min_yield'] is not None and 'yield' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['yield'] >= self.filter_params['min_yield']]
            self.logger.info(f"After min_yield filter: {len(filtered_df)} bonds")
            
        if self.filter_params['max_yield'] is not None and 'yield' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['yield'] <= self.filter_params['max_yield']]
            self.logger.info(f"After max_yield filter: {len(filtered_df)} bonds")
        
        # Фильтр по дюрации
        if 'duration_years' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['duration_years'] >= self.filter_params['min_duration']) & 
                (filtered_df['duration_years'] <= self.filter_params['max_duration'])
            ]
            self.logger.info(f"After duration filter: {len(filtered_df)} bonds")
        
        # Фильтр по дате погашения
        if self.filter_params['max_expiration_date'] and 'expiration_date' in filtered_df.columns:
            max_exp = pd.to_datetime(self.filter_params['max_expiration_date'])
            filtered_df = filtered_df[pd.to_datetime(filtered_df['expiration_date']) <= max_exp]
            self.logger.info(f"After max_expiration_date filter: {len(filtered_df)} bonds")
        
        # Фильтр по ликвидности
        if self.filter_params['min_liquidity'] is not None:
            if 'trading_volume' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['trading_volume'] >= self.filter_params['min_liquidity']]
                self.logger.info(f"After min_liquidity filter: {len(filtered_df)} bonds")
        
        # Исключение эмитентов
        if self.filter_params['excluded_issuers']:
            for column in ['full_name', 'name']:
                if column in filtered_df.columns:
                    before = len(filtered_df)
                    for issuer in self.filter_params['excluded_issuers']:
                        filtered_df = filtered_df[~filtered_df[column].str.contains(issuer, case=False, na=False, regex=True)]
                    self.logger.info(f"After excluded_issuers filter: {len(filtered_df)} bonds (-{before-len(filtered_df)})")
                    break
        
        # Исключение облигаций
        if self.filter_params['excluded_bonds']:
            for column in ['security_code', 'secid']:
                if column in filtered_df.columns:
                    before = len(filtered_df)
                    filtered_df = filtered_df[~filtered_df[column].isin(self.filter_params['excluded_bonds'])]
                    self.logger.info(f"After excluded_bonds filter: {len(filtered_df)} bonds (-{before-len(filtered_df)})")
                    break
        
        # Включение только указанных облигаций
        if self.filter_params['include_only']:
            for column in ['security_code', 'secid']:
                if column in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[column].isin(self.filter_params['include_only'])]
                    self.logger.info(f"After include_only filter: {len(filtered_df)} bonds")
                    break
        
        # Удаление дубликатов
        for column in ['security_code', 'secid']:
            if column in filtered_df.columns:
                filtered_df = filtered_df.drop_duplicates(subset=[column], keep='last')
                self.logger.info(f"After removing duplicates: {len(filtered_df)} bonds")
                break
                
        # Удаление строк с пропущенными значениями в критичных колонках
        for col in ['yield', 'duration_years', 'price_pct']:
            if col in filtered_df.columns:
                before = len(filtered_df)
                filtered_df = filtered_df.dropna(subset=[col])
                self.logger.info(f"Removed {before - len(filtered_df)} bonds with missing {col}")
        
        self.logger.info(f"Final filtered dataset: {len(filtered_df)} bonds from initial {initial_count}")
        return filtered_df
            
    def load_and_process_all_bonds(self, 
                                  date_range=None,
                                  use_latest_snapshot_only=True,
                                  min_yield=6.0,
                                  max_yield=None,
                                  min_duration=0.1,
                                  max_duration=3.0,
                                  max_expiration_date=None,
                                  min_liquidity=None,
                                  excluded_issuers=None,
                                  excluded_bonds=None,
                                  include_only=None):
        """Load and process all bond files with filter criteria"""
        # Если указан прямой файл с данными, используем его
        if self.direct_data_file:
            return self.process_direct_data(
                min_yield=min_yield,
                max_yield=max_yield,
                min_duration=min_duration,
                max_duration=max_duration,
                max_expiration_date=max_expiration_date,
                min_liquidity=min_liquidity,
                excluded_issuers=excluded_issuers,
                excluded_bonds=excluded_bonds,
                include_only=include_only,
                use_latest_snapshot_only=use_latest_snapshot_only
            )
        
        self.logger.info(f"Starting to load and process bond data from {self.bonds_dir}")
        bond_snapshots = []
        
        # Store the filter parameters
        self.filter_params = {
            'date_range': date_range,
            'use_latest_snapshot_only': use_latest_snapshot_only,
            'min_yield': min_yield,
            'max_yield': max_yield,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'max_expiration_date': max_expiration_date,
            'min_liquidity': min_liquidity,
            'excluded_issuers': excluded_issuers or [],
            'excluded_bonds': excluded_bonds or [],
            'include_only': include_only
        }
        
        for file in self.bond_files:
            try:
                file_path = os.path.join(self.bonds_dir, file)
                
                # Extract date from filename
                match = re.search(r'(\d{4}-\d{2}-\d{2})', file)
                if match:
                    date_str = match.group(1)
                    snapshot_date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    # If no date in filename, try to get from file modification time
                    mod_time = os.path.getmtime(file_path)
                    snapshot_date = datetime.fromtimestamp(mod_time).replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Filter by date range if specified
                if date_range:
                    start_date = datetime.strptime(date_range[0], '%Y-%m-%d') if isinstance(date_range[0], str) else date_range[0]
                    end_date = datetime.strptime(date_range[1], '%Y-%m-%d') if isinstance(date_range[1], str) else date_range[1]
                    if snapshot_date < start_date or snapshot_date > end_date:
                        continue
                
                # Read the bond data - try different separators
                try:
                    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                except:
                    try:
                        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
                    except:
                        self.logger.warning(f"Could not read file {file} with standard separators")
                        continue
                
                # Check if this is the expected format
                if ('secid' in df.columns and 'yield' in df.columns) or \
                   ('security_code' in df.columns and 'yield' in df.columns):
                    # This is the expected format
                    df['snapshot_date'] = snapshot_date
                    
                    # Standardize column names
                    df = self._standardize_moex_columns(df)
                    
                    # Clean data types
                    df = self._clean_data_types_moex(df)
                    
                    bond_snapshots.append(df)
                    self.logger.info(f"Successfully loaded file {file} with {len(df)} bonds from {snapshot_date.strftime('%Y-%m-%d')}")
                else:
                    self.logger.warning(f"Skipping file with unexpected format: {file}")
                    
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}\n{traceback.format_exc()}")
                continue
                
        if not bond_snapshots:
            self.logger.error("No bond data found!")
            return pd.DataFrame()
            
        # Combine all snapshots
        try:
            all_bonds = pd.concat(bond_snapshots, ignore_index=True)
            self.logger.info(f"Combined data for {len(all_bonds)} bond entries")
        except Exception as e:
            self.logger.error(f"Error combining data: {e}")
            return pd.DataFrame()
        
        # Use only the latest snapshot if needed
        if use_latest_snapshot_only and 'snapshot_date' in all_bonds.columns:
            latest_date = all_bonds['snapshot_date'].max()
            all_bonds = all_bonds[all_bonds['snapshot_date'] == latest_date]
            self.logger.info(f"Applied filter: only latest snapshot from {latest_date.strftime('%Y-%m-%d')} ({len(all_bonds)} entries)")
        
        # Apply filters
        filtered_bonds = self._apply_filters(all_bonds)
        
        # Calculate bond metrics
        self.processed_bonds = self.calculate_bond_metrics(filtered_bonds)
        self.logger.info(f"Completed processing {len(self.processed_bonds)} bonds")
        
        # Detect market condition
        self.market_condition = self._detect_market_condition()
        self.logger.info(f"Detected market condition: {self.market_condition}")
        
        return self.processed_bonds
    
    def _standardize_moex_columns(self, df):
        """Standardize column names from MOEX format"""
        # Create a mapping for column names
        column_mapping = {
            'secid': 'security_code',
            'name': 'full_name',
            'price': 'price_pct',
            'tradedate': 'trade_date',
            'yield': 'yield',
            'duration': 'duration_months',
            'maturity_date': 'expiration_date',
            'volume': 'trading_volume',
            'value': 'trading_value',
            'coupon_percent': 'coupon_rate',
            'coupon_value': 'coupon_value',
            'face_value': 'nominal_value',
            'face_unit': 'currency',
            'accint': 'accrued_interest'
        }
        
        # Rename only existing columns
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)
        
        # Log transformed columns
        self.logger.info(f"Standardized columns: {list(rename_cols.values())}")
        
        return df
        
    def _clean_data_types_moex(self, df):
        """Clean and convert data types for MOEX format"""
        # Convert numeric columns
        numeric_cols = ['price_pct', 'yield', 'duration_months', 'trading_volume', 
                       'trading_value', 'coupon_rate', 'coupon_value', 'nominal_value']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['expiration_date', 'trade_date', 'offer_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert duration from months to years if it exists
        if 'duration_months' in df.columns:
            # In MOEX data, duration is often already in years
            if df['duration_months'].max() < 100:  # If max duration < 100, it's likely already in years
                df['duration_years'] = df['duration_months']
            else:
                df['duration_years'] = df['duration_months'] / 12
                
        # Check if we need to calculate tax benefit (not explicitly provided in MOEX data)
        if 'tax_benefit' not in df.columns:
            # Add a placeholder. In real implementation, this would need proper logic
            df['tax_benefit'] = False
            # Could use logic like: df['tax_benefit'] = df['full_name'].str.contains('ОФЗ|Минфин')
            
        self.logger.info("Completed data type conversion")
        
        return df
        
    def _apply_filters(self, df):
        """Apply all filters to the data"""
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Filter by yield
        if self.filter_params['min_yield'] is not None and 'yield' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['yield'] >= self.filter_params['min_yield']]
            self.logger.info(f"Applied min_yield filter ({self.filter_params['min_yield']}): {len(filtered_df)} bonds remaining")
            
        if self.filter_params['max_yield'] is not None and 'yield' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['yield'] <= self.filter_params['max_yield']]
            self.logger.info(f"Applied max_yield filter ({self.filter_params['max_yield']}): {len(filtered_df)} bonds remaining")
        
        # Filter by duration
        if 'duration_years' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['duration_years'] >= self.filter_params['min_duration']) & 
                (filtered_df['duration_years'] <= self.filter_params['max_duration'])
            ]
            self.logger.info(f"Applied duration filter ({self.filter_params['min_duration']}-{self.filter_params['max_duration']} years): {len(filtered_df)} bonds remaining")
        
        # Filter by expiration date
        if self.filter_params['max_expiration_date'] and 'expiration_date' in filtered_df.columns:
            max_exp = pd.to_datetime(self.filter_params['max_expiration_date'])
            filtered_df = filtered_df[filtered_df['expiration_date'] <= max_exp]
            self.logger.info(f"Applied max_expiration_date filter ({max_exp.strftime('%Y-%m-%d')}): {len(filtered_df)} bonds remaining")
        
        # Filter by liquidity
        if self.filter_params['min_liquidity'] is not None and 'trading_volume' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['trading_volume'] >= self.filter_params['min_liquidity']]
            self.logger.info(f"Applied min_liquidity filter ({self.filter_params['min_liquidity']}): {len(filtered_df)} bonds remaining")
            
        # Exclude specific issuers
        if self.filter_params['excluded_issuers'] and 'full_name' in filtered_df.columns:
            for issuer in self.filter_params['excluded_issuers']:
                before = len(filtered_df)
                filtered_df = filtered_df[~filtered_df['full_name'].str.contains(issuer, case=False, na=False)]
                self.logger.info(f"Excluded issuer '{issuer}': removed {before - len(filtered_df)} bonds")
        
        # Exclude specific bonds
        if self.filter_params['excluded_bonds'] and 'security_code' in filtered_df.columns:
            before = len(filtered_df)
            filtered_df = filtered_df[~filtered_df['security_code'].isin(self.filter_params['excluded_bonds'])]
            self.logger.info(f"Excluded specified bonds: removed {before - len(filtered_df)}")
            
        # Include only specified bonds
        if self.filter_params['include_only'] and 'security_code' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['security_code'].isin(self.filter_params['include_only'])]
            self.logger.info(f"Included only specified bonds: {len(filtered_df)} remaining")
        
        # Remove duplicates by security code
        if 'security_code' in filtered_df.columns:
            filtered_df = filtered_df.drop_duplicates(subset=['security_code'], keep='last')
            
        # Remove bonds with missing critical data
        for col in ['yield', 'duration_years', 'price_pct']:
            if col in filtered_df.columns:
                before = len(filtered_df)
                filtered_df = filtered_df.dropna(subset=[col])
                self.logger.info(f"Removed {before - len(filtered_df)} bonds with missing {col}")
                
        self.logger.info(f"After applying all filters: {len(filtered_df)} bonds remaining from {initial_count}")
        return filtered_df
    
    def _detect_market_condition(self, market_condition=None):
        """Detect market condition based on bond data"""
        if market_condition is not None:
            return market_condition
            
        if self.processed_bonds is None or 'yield' not in self.processed_bonds.columns:
            return 'neutral'
            
        avg_yield = self.processed_bonds['yield'].mean()
        
        # Conditional logic
        if avg_yield > 12:
            return 'high_rate'
        elif avg_yield < 8:
            return 'low_rate'
        else:
            return 'neutral'
    
    def calculate_bond_metrics(self, bonds_df, score_weights=None):
        """Calculate additional bond metrics"""
        if score_weights is None:
            score_weights = {
                'yield': 0.5,
                'duration': 0.3,
                'tax_benefit': 0.1,
                'price_discount': 0.1
            }
            
        df = bonds_df.copy()
        
        try:
            # Calculate risk-adjusted yield (yield / duration)
            if 'yield' in df.columns and 'duration_years' in df.columns:
                df['risk_adjusted_yield'] = df['yield'] / df['duration_years'].clip(lower=0.1)
            
            # Calculate z-scores for yields
            if 'yield' in df.columns and 'snapshot_date' in df.columns:
                df['yield_zscore'] = df.groupby('snapshot_date')['yield'].transform(
                    lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
                )
            elif 'yield' in df.columns:
                df['yield_zscore'] = (df['yield'] - df['yield'].mean()) / df['yield'].std() if df['yield'].std() > 0 else 0
            
            # Calculate price deviation from par (100%)
            if 'price_pct' in df.columns:
                df['price_deviation'] = df['price_pct'] - 100
            
            # Calculate bond score with customizable weights
            score_components = []
            
            if 'yield_zscore' in df.columns and 'yield' in score_weights:
                score_components.append(df['yield_zscore'] * score_weights['yield'])
                
            if 'duration_years' in df.columns and 'duration' in score_weights:
                score_components.append(
                    (1 / df['duration_years'].clip(0.5, None)) * score_weights['duration']
                )
                
            if 'tax_benefit' in df.columns and 'tax_benefit' in score_weights:
                score_components.append(
                    df['tax_benefit'].astype(float) * score_weights['tax_benefit']
                )
                            
            if 'price_deviation' in df.columns and 'price_discount' in score_weights:
                score_components.append(
                    (df['price_deviation'] < -1).astype(float) * score_weights['price_discount']
                )
            
            if score_components:
                # Sum all score components
                df['bond_score'] = pd.concat(score_components, axis=1).sum(axis=1)
            else:
                df['bond_score'] = 0.5  # Default neutral score
            
            # Calculate time to maturity in years
            if 'expiration_date' in df.columns and 'snapshot_date' in df.columns:
                # Убедимся, что оба столбца имеют тип datetime
                if not pd.api.types.is_datetime64_any_dtype(df['expiration_date']):
                    df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
                
                if not pd.api.types.is_datetime64_any_dtype(df['snapshot_date']):
                    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], errors='coerce')
                
                # Теперь выполняем вычитание
                df['years_to_maturity'] = (df['expiration_date'] - df['snapshot_date']).dt.days / 365.25
            
            # Calculate liquidity score (if volume data available)
            if 'trading_volume' in df.columns:
                df['liquidity_score'] = df['trading_volume'].rank(pct=True)
            
            self.logger.info(f"Calculated metrics for {len(df)} bonds")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}\n{traceback.format_exc()}")
            return bonds_df

    
    def get_bonds_for_portfolio(self, n_bonds=5, weighting_strategy='inverse_duration', portfolio_stability=0.7):
        """Get top recommended bonds for portfolio inclusion"""
        if self.processed_bonds is None or len(self.processed_bonds) == 0:
            self.logger.warning("No bond data available for portfolio formation")
            return pd.DataFrame()
                    
        try:
            # Get latest data
            if 'snapshot_date' in self.processed_bonds.columns:
                latest_date = self.processed_bonds['snapshot_date'].max()
                bond_pool = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
            else:
                bond_pool = self.processed_bonds.copy()
            
            if len(bond_pool) == 0:
                self.logger.warning("No suitable bonds for portfolio")
                return pd.DataFrame()
            
            self.logger.info(f"Available {len(bond_pool)} bonds for portfolio formation")
            
            # Select more bonds than needed (for stability considerations)
            expanded_n = min(int(n_bonds * 1.5) + 2, len(bond_pool))
            
            # Select best bonds by score
            if 'bond_score' in bond_pool.columns:
                top_candidates = bond_pool.nlargest(expanded_n, 'bond_score')
            else:
                top_candidates = bond_pool.head(expanded_n)
            
            # Apply portfolio stability logic
            top_bonds = self._apply_portfolio_stability(top_candidates, n_bonds, portfolio_stability)
            
            # Create summary for portfolio inclusion
            columns_to_keep = ['security_code', 'full_name', 'yield', 'duration_years', 
                             'risk_adjusted_yield', 'bond_score', 'price_pct', 'currency']
            columns_to_keep = [col for col in columns_to_keep if col in top_bonds.columns]
            
            bond_summary = top_bonds[columns_to_keep].copy()
            
            # Calculate weights based on strategy
            bond_summary = self._calculate_weights(bond_summary, weighting_strategy)
            
            # Add market condition info
            bond_summary.attrs['market_condition'] = self.market_condition
            bond_summary.attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save current recommendations for future comparison
            self._save_current_recommendations(bond_summary)
            
            self.logger.info(f"Portfolio formed with {len(bond_summary)} bonds")
            return bond_summary
            
        except Exception as e:
            self.logger.error(f"Error forming portfolio: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()
    
    def _apply_portfolio_stability(self, candidates, n_bonds, stability=0.7):
        """Apply portfolio stability logic to minimize rebalancing"""
        if not self.previous_recommendations or 'bonds' not in self.previous_recommendations:
            self.logger.info("No previous recommendations - using only current data")
            return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
        
        try:
            # Get previous recommendations
            prev_bonds = self.previous_recommendations['bonds']
            prev_codes = [b['security_code'] for b in prev_bonds if 'security_code' in b]
            
            # Determine how many bonds to keep
            keep_n = int(n_bonds * stability)
            
            # Find bonds that were in previous portfolio and are in current candidates
            kept_bonds = candidates[candidates['security_code'].isin(prev_codes)]
            
            if len(kept_bonds) > 0:
                # Sort kept bonds by score
                kept_bonds = kept_bonds.nlargest(min(keep_n, len(kept_bonds)), 'bond_score') if 'bond_score' in kept_bonds.columns else kept_bonds.head(min(keep_n, len(kept_bonds)))
                
                # Select new bonds, excluding those already kept
                new_n = n_bonds - len(kept_bonds)
                new_bonds = candidates[~candidates['security_code'].isin(kept_bonds['security_code'])].nlargest(new_n, 'bond_score') if 'bond_score' in candidates.columns else candidates[~candidates['security_code'].isin(kept_bonds['security_code'])].head(new_n)
                
                # Combine kept and new bonds
                result = pd.concat([kept_bonds, new_bonds])
                self.logger.info(f"Kept {len(kept_bonds)} bonds from previous portfolio, added {len(new_bonds)} new ones")
                
                return result
            else:
                self.logger.info("No overlaps with previous portfolio - using only new data")
                return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
                
        except Exception as e:
            self.logger.warning(f"Error applying portfolio stability: {e} - using only current data")
            return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
    
    def _calculate_weights(self, bond_summary, weighting_strategy='inverse_duration'):
        """Calculate weights based on selected strategy"""
        try:
            if weighting_strategy == 'equal':
                # Equal weights
                bond_summary['weight'] = 1.0 / len(bond_summary)
                
            elif weighting_strategy == 'inverse_duration' and 'duration_years' in bond_summary.columns:
                # Weights inversely proportional to duration
                inverse_duration = 1 / bond_summary['duration_years'].clip(lower=0.1)
                bond_summary['weight'] = inverse_duration / inverse_duration.sum()
                
            elif weighting_strategy == 'yield' and 'yield' in bond_summary.columns:
                # Weights proportional to yield
                bond_summary['weight'] = bond_summary['yield'] / bond_summary['yield'].sum()
                
            elif weighting_strategy == 'bond_score' and 'bond_score' in bond_summary.columns:
                # Weights proportional to bond score
                scores = bond_summary['bond_score'].clip(lower=0.01)  # Avoid negative values
                bond_summary['weight'] = scores / scores.sum()
                
            else:
                # Default to equal weights
                bond_summary['weight'] = 1.0 / len(bond_summary)
            
            # Round weights for readability
            bond_summary['weight'] = bond_summary['weight'].round(4)
            self.logger.info(f"Calculated weights using '{weighting_strategy}' strategy")
            
            return bond_summary
        
        except Exception as e:
            self.logger.error(f"Error calculating weights: {e} - using equal weights")
            bond_summary['weight'] = 1.0 / len(bond_summary)
            return bond_summary
    
    def _save_current_recommendations(self, bond_summary):
        """Save current recommendations for future comparison"""
        try:
            results_dir = os.path.join(os.path.dirname(self.bonds_dir), 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            recommendations = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_condition': self.market_condition,
                'bonds': bond_summary.to_dict(orient='records')
            }
            
            # Save to JSON for future use
            with open(os.path.join(results_dir, 'latest_recommendations.json'), 'w') as f:
                json.dump(recommendations, f, indent=2)
                
            self.logger.info("Current recommendations saved for future use")
        except Exception as e:
            self.logger.error(f"Error saving recommendations: {e}")
    
    def visualize_bond_universe(self, output_dir=None):
        """Create visualization of the bond universe"""
        if self.processed_bonds is None or len(self.processed_bonds) == 0:
            self.logger.warning("No data available for visualization")
            return None
        
        try:
            # Create directory for visualizations
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(self.bonds_dir), 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            # Get data for visualization
            if 'snapshot_date' in self.processed_bonds.columns:
                latest_date = self.processed_bonds['snapshot_date'].max()
                plot_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
                date_title = f"{latest_date.strftime('%Y-%m-%d')}" if hasattr(latest_date, 'strftime') else "Latest date"
            else:
                plot_bonds = self.processed_bonds.copy()
                date_title = "All available data"
            
            # Create yield vs duration visualization
            if 'duration_years' in plot_bonds.columns and 'yield' in plot_bonds.columns:
                plt.figure(figsize=(12, 8))
                
                # Main scatter plot - используем параметры в зависимости от доступных данных
                scatter_params = {
                    'alpha': 0.7,
                    's': 100
                }
                
                # Добавляем параметры для цветового отображения только если есть bond_score
                if 'bond_score' in plot_bonds.columns and not plot_bonds['bond_score'].isna().all():
                    scatter_params['c'] = plot_bonds['bond_score']
                    scatter_params['cmap'] = 'viridis'
                
                scatter = plt.scatter(
                    plot_bonds['duration_years'], 
                    plot_bonds['yield'], 
                    **scatter_params
                )
                
                # Highlight different currencies if available
                if 'currency' in plot_bonds.columns:
                    currencies = plot_bonds['currency'].unique()
                    if len(currencies) > 1:
                        for i, currency in enumerate(currencies):
                            currency_bonds = plot_bonds[plot_bonds['currency'] == currency]
                            if not currency_bonds.empty:
                                plt.scatter(
                                    currency_bonds['duration_years'],
                                    currency_bonds['yield'],
                                    alpha=0.5, s=120,
                                    edgecolors='red' if i == 0 else ('blue' if i == 1 else 'green'),
                                    facecolors='none',
                                    linewidth=2,
                                    label=f'{currency}'
                                )
                        plt.legend()
                
                # Annotate top bonds
                if 'bond_score' in plot_bonds.columns and len(plot_bonds) > 0:
                    # Берем не больше 5 облигаций, но не больше имеющихся
                    top_count = min(5, len(plot_bonds))
                    if top_count > 0:
                        top_bonds = plot_bonds.nlargest(top_count, 'bond_score')
                        for _, bond in top_bonds.iterrows():
                            if pd.notna(bond['duration_years']) and pd.notna(bond['yield']):
                                plt.annotate(
                                    bond['security_code'],
                                    (bond['duration_years'], bond['yield']),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=9,
                                    fontweight='bold'
                                )
                
                if 'c' in scatter_params:
                    plt.colorbar(scatter, label='Bond Score')
                    
                plt.title(f'Bond Universe: Yield vs Duration - {date_title}')
                plt.xlabel('Duration (years)')
                plt.ylabel('Yield (%)')
                plt.grid(True, alpha=0.3)
                
                output_file = os.path.join(output_dir, f'bond_universe_{datetime.now().strftime("%Y%m%d")}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Visualization saved to {output_file}")
                return output_file
            else:
                self.logger.warning("Missing required columns for visualization")
                return None
        
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}\n{traceback.format_exc()}")
            return None

            
    def generate_portfolio_analysis(self, portfolio, output_dir=None):
        """Generate detailed portfolio analysis and visualizations"""
        if portfolio is None or len(portfolio) == 0:
            self.logger.warning("No portfolio data for analysis")
            return None
            
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.bonds_dir), 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Calculate portfolio characteristics
            portfolio_stats = {
                'number_of_bonds': len(portfolio),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            if 'yield' in portfolio.columns and 'weight' in portfolio.columns:
                portfolio_stats['weighted_yield'] = (portfolio['yield'] * portfolio['weight']).sum()
                
            if 'duration_years' in portfolio.columns and 'weight' in portfolio.columns:
                portfolio_stats['weighted_duration'] = (portfolio['duration_years'] * portfolio['weight']).sum()
                
            if 'bond_score' in portfolio.columns:
                portfolio_stats['avg_score'] = portfolio['bond_score'].mean()
            
            # Create portfolio weights chart
            plt.figure(figsize=(10, 6))
            plt.bar(portfolio['security_code'], portfolio['weight'], color='teal', alpha=0.7)
            plt.title('Bond Portfolio Weights')
            plt.ylabel('Weight')
            plt.xlabel('Bond')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the chart
            weights_file = os.path.join(output_dir, f'portfolio_weights_{datetime.now().strftime("%Y%m%d")}.png')
            plt.savefig(weights_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create risk/return plot
            if 'duration_years' in portfolio.columns and 'yield' in portfolio.columns:
                plt.figure(figsize=(10, 6))
                
                # If we have the original bond universe, plot it as background
                if self.processed_bonds is not None and len(self.processed_bonds) > 0:
                    plt.scatter(
                        self.processed_bonds['duration_years'], 
                        self.processed_bonds['yield'],
                        alpha=0.3, s=30, color='gray', label='All bonds'
                    )
                
                # Plot portfolio bonds
                plt.scatter(
                    portfolio['duration_years'],
                    portfolio['yield'],
                    alpha=0.8, s=100, color='blue', label='Portfolio bonds'
                )
                
                # Annotate each bond
                for _, bond in portfolio.iterrows():
                    plt.annotate(
                        bond['security_code'],
                        (bond['duration_years'], bond['yield']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold'
                    )
                
                plt.title('Portfolio Risk/Return Profile')
                plt.xlabel('Duration (years) - Risk')
                plt.ylabel('Yield (%) - Return')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save the chart
                risk_return_file = os.path.join(output_dir, f'portfolio_risk_return_{datetime.now().strftime("%Y%m%d")}.png')
                plt.savefig(risk_return_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save portfolio statistics
            with open(os.path.join(output_dir, f'portfolio_stats_{datetime.now().strftime("%Y%m%d")}.json'), 'w') as f:
                json.dump(portfolio_stats, f, indent=2)
            
            self.logger.info("Portfolio analysis generated successfully")
            return portfolio_stats
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio analysis: {e}\n{traceback.format_exc()}")
            return None
        
def run_pipeline_bonds_processor(
    base_path,
    dataset_path=None,
    results_dir=None,
    min_yield=6.0,
    max_yield=22.0,
    min_duration=0.1,
    max_duration=5.0,
    excluded_issuers=None,
    use_latest_snapshot_only=True,
    n_bonds=5,
    weighting_strategy='inverse_duration',
    portfolio_stability=0.7
):
    """
    Полный пайплайн для обработки облигаций и создания оптимального портфеля.
    
    Parameters:
    -----------
    base_path : str
        Базовый путь к проекту
    dataset_path : str, optional
        Путь к файлу с датасетом облигаций
    results_dir : str, optional
        Директория для сохранения результатов
    min_yield : float, default=6.0
        Минимальная доходность для фильтрации
    max_yield : float, default=22.0
        Максимальная доходность для фильтрации
    min_duration : float, default=0.1
        Минимальная дюрация для фильтрации
    max_duration : float, default=5.0
        Максимальная дюрация для фильтрации
    excluded_issuers : list, optional
        Список эмитентов для исключения
    use_latest_snapshot_only : bool, default=True
        Использовать только последний снапшот
    n_bonds : int, default=5
        Количество облигаций в портфеле
    weighting_strategy : str, default='inverse_duration'
        Стратегия взвешивания портфеля
    portfolio_stability : float, default=0.7
        Коэффициент стабильности портфеля
        
    Returns:
    --------
    dict
        Результаты выполнения пайплайна, включая портфель и пути к сохраненным файлам
    """
    import os
    import sys
    from datetime import datetime
    
    if excluded_issuers is None:
        excluded_issuers = ['ВТБ', 'Мечел']
    
    # Установка путей
    if dataset_path is None:
        dataset_path = f"{base_path}/data/processed_data/BONDS/moex/threshold_dataset_20240101_20250422_99/bonds_dataset.csv"
    
    if results_dir is None:
        results_dir = f"{base_path}/data/processed_data/BONDS/results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Импорт BondsProcessor - гибкий подход к импорту
    try:
        # Пробуем импортировать напрямую
        from bonds_processor import BondsProcessor
    except ImportError:
        try:
            # Через полный путь
            from pys.data_collection.bonds_processor import BondsProcessor
        except ImportError:
            # Добавляем базовый путь и пробуем снова
            if base_path not in sys.path:
                sys.path.append(base_path)
            try:
                from bonds_processor import BondsProcessor
            except ImportError:
                sys.path.append(os.path.join(base_path, 'pys', 'data_collection'))
                from bonds_processor import BondsProcessor
    
    processor = BondsProcessor(
        bonds_dir=f"{base_path}/data/processed_data/BONDS/processed",
        direct_data_file=dataset_path
    )
    
    # Обработка облигаций
    processed_bonds = processor.load_and_process_all_bonds(
        min_yield=min_yield,
        max_yield=max_yield,
        min_duration=min_duration,
        max_duration=max_duration,
        excluded_issuers=excluded_issuers,
        use_latest_snapshot_only=use_latest_snapshot_only
    )
    
    results = {
        'success': False,
        'processed_bonds': None,
        'portfolio': None,
        'portfolio_path': None,
        'visualization_path': None,
        'stats': None
    }
    
    if processed_bonds is not None and not processed_bonds.empty:
        print(f"Обработано {len(processed_bonds)} облигаций")
        results['processed_bonds'] = processed_bonds
        results['success'] = True
        
        # Формирование портфеля
        portfolio = processor.get_bonds_for_portfolio(
            n_bonds=n_bonds,
            weighting_strategy=weighting_strategy,
            portfolio_stability=portfolio_stability
        )
        
        if portfolio is not None and not portfolio.empty:
            results['portfolio'] = portfolio
            print("\n=== Рекомендованный облигационный портфель ===")
            
            # Определяем колонки для вывода
            display_cols = [col for col in ['security_code', 'full_name', 'yield', 'duration_years', 'weight'] 
                        if col in portfolio.columns]
            
            # Выводим портфель
            print(portfolio[display_cols])
            
            # Сохраняем портфель
            portfolio_path = os.path.join(results_dir, f"bond_portfolio_{datetime.now().strftime('%Y%m%d')}.csv")
            portfolio.to_csv(portfolio_path, index=False)
            print(f"\nПортфель сохранен в {portfolio_path}")
            results['portfolio_path'] = portfolio_path
            
            # Визуализация
            viz_path = processor.visualize_bond_universe(output_dir=results_dir)
            if viz_path:
                print(f"Визуализация сохранена в {viz_path}")
                results['visualization_path'] = viz_path
                
            # Генерируем анализ портфеля
            stats = processor.generate_portfolio_analysis(portfolio, output_dir=results_dir)
            if stats:
                results['stats'] = stats
                print("\n=== Характеристики портфеля ===")
                print(f"Средневзвешенная доходность: {stats.get('weighted_yield', 0):.2f}%")
                print(f"Средневзвешенная дюрация: {stats.get('weighted_duration', 0):.2f} лет")
                
                risk_level = 'Низкий'
                if stats.get('weighted_duration', 0) > 2:
                    risk_level = 'Средний'
                if stats.get('weighted_duration', 0) > 4:
                    risk_level = 'Высокий'
                    
                print(f"Уровень риска (дюрация): {risk_level}")
                
            print("\nАнализ успешно завершен!")
        else:
            print("Не удалось сформировать портфель")
    else:
        print("Не удалось обработать данные облигаций")
    
    return results
