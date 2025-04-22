# bonds_processor.py
import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class BondsProcessor:
    def __init__(self, bonds_dir="processed_data/bonds_csv"):
        self.bonds_dir = bonds_dir
        self.bond_files = sorted([f for f in os.listdir(bonds_dir) if f.endswith('.csv')])
        self.processed_bonds = None
        
    def load_and_process_all_bonds(self):
        """Load all bond CSV files and merge them into a time series dataset"""
        bond_snapshots = []
        
        for file in self.bond_files:
            file_path = os.path.join(self.bonds_dir, file)
            # Extract date from filename (assuming format: bond_search_YYYY-MM-DD.csv)
            date_str = file.replace('bond_search_', '').replace('.csv', '')
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Read the bond data with the correct encoding and column names
            df = pd.read_csv(file_path)
            df['snapshot_date'] = date
            bond_snapshots.append(df)
            
        # Combine all snapshots
        all_bonds = pd.concat(bond_snapshots, ignore_index=True)
        
        # Rename columns to more accessible format
        all_bonds.columns = [
            'full_name',
            'security_code',
            'price_pct',
            'trading_volume',
            'yield',
            'duration_months',
            'tax_benefit',
            'snapshot_date'
        ]
        
        # Clean and convert data types
        all_bonds['price_pct'] = pd.to_numeric(all_bonds['price_pct'], errors='coerce')
        all_bonds['yield'] = pd.to_numeric(all_bonds['yield'], errors='coerce')
        all_bonds['duration_months'] = pd.to_numeric(all_bonds['duration_months'], errors='coerce')
        
        # Convert duration from months to years
        all_bonds['duration_years'] = all_bonds['duration_months'] / 12
        
        # Convert tax benefit to boolean if it's not already
        if all_bonds['tax_benefit'].dtype == 'object':
            all_bonds['tax_benefit'] = all_bonds['tax_benefit'].map({'True': True, 'False': False})
        
        # Calculate key bond metrics
        self.processed_bonds = self.calculate_bond_metrics(all_bonds)
        return self.processed_bonds
    
    def calculate_bond_metrics(self, bonds_df):
        """Calculate additional bond metrics like risk-adjusted yield, etc."""
        df = bonds_df.copy()
        
        # Calculate risk-adjusted yield (yield / duration)
        df['risk_adjusted_yield'] = df['yield'] / df['duration_years']
        
        # Calculate z-scores for yields within each snapshot date
        df['yield_zscore'] = df.groupby('snapshot_date')['yield'].transform(
            lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
        )
        
        # Calculate price change from nominal (100%)
        df['price_deviation'] = df['price_pct'] - 100
        
        # Calculate bond score (higher is better)
        # Components: high yield, short duration, tax benefit, price discount
        df['bond_score'] = (
            df['yield_zscore'] * 0.5 +  # Higher yield is better
            (1 / df['duration_years'].clip(0.5, None)) * 0.3 +  # Shorter duration is better (with min of 0.5 year)
            df['tax_benefit'].astype(int) * 0.1 +  # Tax benefit adds 10%
            (df['price_deviation'] < -1).astype(int) * 0.1  # Price discount adds 10%
        )
        
        return df
    
    def select_risk_free_candidates(self, min_yield=7.0, max_duration=3.0):
        """Select bonds suitable for the risk-free portion of portfolio"""
        if self.processed_bonds is None:
            self.load_and_process_all_bonds()
            
        # Get most recent snapshot
        latest_date = self.processed_bonds['snapshot_date'].max()
        latest_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
        
        # Filter by yield and duration
        risk_free_bonds = latest_bonds[
            (latest_bonds['yield'] >= min_yield) & 
            (latest_bonds['duration_years'] <= max_duration)
        ]
        
        # Sort by risk-adjusted yield (descending)
        risk_free_bonds = risk_free_bonds.sort_values('risk_adjusted_yield', ascending=False)
        
        # Add percentile rank for final selection
        risk_free_bonds['risk_free_rank'] = risk_free_bonds['bond_score'].rank(pct=True)
        
        return risk_free_bonds
    
    def get_bonds_for_portfolio(self, n_bonds=5):
        """Get top recommended bonds for portfolio inclusion"""
        rf_candidates = self.select_risk_free_candidates()
        
        # Select top bonds by bond score
        top_bonds = rf_candidates.nlargest(n_bonds, 'bond_score')
        
        # Create a summary for portfolio inclusion
        bond_summary = top_bonds[['security_code', 'full_name', 'yield', 
                                 'duration_years', 'risk_adjusted_yield', 'bond_score']].copy()
        
        # Calculate recommended weights inversely proportional to duration
        inverse_duration = 1 / bond_summary['duration_years']
        bond_summary['weight'] = inverse_duration / inverse_duration.sum()
        
        return bond_summary
    
    def visualize_bond_universe(self):
        """Create visualization of the bond universe"""
        if self.processed_bonds is None:
            self.load_and_process_all_bonds()
            
        # Get the most recent snapshot
        latest_date = self.processed_bonds['snapshot_date'].max()
        latest_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
        
        # Create yield curve visualization
        plt.figure(figsize=(12, 8))
        
        # Plot yield vs duration
        scatter = plt.scatter(
            latest_bonds['duration_years'], 
            latest_bonds['yield'], 
            alpha=0.7, s=100, 
            c=latest_bonds['bond_score'], 
            cmap='viridis'
        )
        
        # Color tax-benefited bonds differently
        tax_benefit_bonds = latest_bonds[latest_bonds['tax_benefit']]
        plt.scatter(
            tax_benefit_bonds['duration_years'], 
            tax_benefit_bonds['yield'], 
            alpha=0.5, s=130, 
            edgecolors='red', facecolors='none', 
            linewidth=2, 
            label='Tax Benefit'
        )
        
        # Annotate top bonds
        top_bonds = latest_bonds.nlargest(5, 'bond_score')
        for _, bond in top_bonds.iterrows():
            plt.annotate(
                bond['security_code'],
                (bond['duration_years'], bond['yield']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.colorbar(scatter, label='Bond Score')
        plt.title(f'Bond Universe: Yield vs Duration ({latest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Duration (Years)')
        plt.ylabel('Yield (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        base_dir = '/Users/aeshef/Documents/GitHub/kursach/data/processed_data/BONDS'
        
        # Save the visualization
        os.makedirs(f'{base_dir}/visualizations', exist_ok=True)
        plt.savefig(f'{base_dir}/visualizations/bond_yield_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk-return visualization
        plt.figure(figsize=(12, 8))
        
        # Plot yield vs risk-adjusted yield
        plt.scatter(
            latest_bonds['yield'], 
            latest_bonds['risk_adjusted_yield'], 
            alpha=0.7, s=100, 
            c=latest_bonds['duration_years'], 
            cmap='coolwarm'
        )
        
        # Annotate top bonds by risk-adjusted yield
        top_raj_bonds = latest_bonds.nlargest(5, 'risk_adjusted_yield')
        for _, bond in top_raj_bonds.iterrows():
            plt.annotate(
                bond['security_code'],
                (bond['yield'], bond['risk_adjusted_yield']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )
        
        plt.colorbar(label='Duration (Years)')
        plt.title(f'Bond Risk-Return Profile ({latest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Yield (%)')
        plt.ylabel('Risk-Adjusted Yield (Yield/Duration)')
        plt.grid(True, alpha=0.3)
        
        # Save the visualization
        plt.savefig(f'{base_dir}/visualizations/bond_risk_return.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return latest_bonds

    def get_risk_free_rate(self):
        """Extract current risk-free rate from selected bonds"""
        if self.processed_bonds is None:
            self.load_and_process_all_bonds()
            
        # Get most recent snapshot
        latest_date = self.processed_bonds['snapshot_date'].max()
        latest_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date]
        
        # Find shortest duration high quality bonds
        short_bonds = latest_bonds[latest_bonds['duration_years'] <= 1.5]
        
        if len(short_bonds) > 0:
            # Use average of top 3 shortest duration bonds
            return short_bonds.nsmallest(3, 'duration_years')['yield'].mean() / 100
        
        # Fallback - lowest yield from all bonds with shortest maturity
        if len(latest_bonds) > 0:
            return latest_bonds.nsmallest(3, 'duration_years')['yield'].mean() / 100
        
        # Default value if no data available
        return 0.9 # 9% typical risk-free rate for RUB

def run_pipeline_bonds(
    # Основные параметры
    bonds_dir="/Users/aeshef/Documents/GitHub/kursach/data/processed_data/BONDS/csv",
    visualize=True,
    select_top_n=5,
    
    # Параметры фильтрации по датам
    date_range=None,  # ('2022-01-01', '2025-06-30') или None для всех данных
    use_latest_snapshot_only=True,  # По умолчанию используем только последние данные
    
    # Параметры отбора облигаций
    min_yield=6.0,  # Минимальная доходность
    max_yield=None,  # Максимальная доходность (None = без ограничения)
    min_duration=0.1,  # Минимальная дюрация в годах
    max_duration=3.0,  # Максимальная дюрация в годах 
    max_expiration_date=None,  # Максимальная дата экспирации ('YYYY-MM-DD')
    min_liquidity=None,  # Минимальный объем торгов
    
    # Параметры для расчета метрик
    score_weights={
        'yield': 0.5,           # Вес доходности
        'duration': 0.3,        # Вес дюрации (обратной)
        'tax_benefit': 0.1,     # Вес налоговой льготы
        'price_discount': 0.1   # Вес скидки к номиналу
    },
    
    # Параметры для формирования портфеля
    weighting_strategy='inverse_duration',  # 'equal', 'inverse_duration', 'yield', 'bond_score'
    excluded_issuers=None,  # Список эмитентов для исключения
    excluded_bonds=None,    # Список кодов облигаций для исключения
    include_only=None,      # Список кодов облигаций, которые нужно обязательно включить
    
    # Параметры для автоматизации
    output_dir=None,  # Каталог для сохранения результатов
    save_results=True,  # Сохранять результаты в файл
    market_condition=None,  # 'high_rate', 'low_rate', 'neutral', None (авто)
    portfolio_stability=0.7,  # Коэфф. стабильности 0-1: выше = меньше изменений между днями
    
    # Расширенные параметры
    detailed_logging=True,  # Подробное логирование процесса
    comparison_with_previous=True,  # Сравнение с предыдущим днем
    export_format='excel'  # 'excel', 'csv', 'json'
):
    """
    Усовершенствованный пайплайн обработки облигаций для ежедневного использования
    
    Пайплайн разработан для ежедневного запуска с целью получения актуальных рекомендаций 
    по составу портфеля облигаций. Имеет гибкие настройки и различные стратегии взвешивания.
    """
    import pandas as pd
    import os
    import numpy as np
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import logging
    import json
    from pathlib import Path
    import traceback
    
    # Настройка логирования
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(bonds_dir), 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'bonds_pipeline_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO if detailed_logging else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('bonds_pipeline')
    logger.info(f"Запуск пайплайна обработки облигаций в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Вспомогательная функция для определения рыночных условий
    def detect_market_condition(bonds_df):
        if market_condition is not None:
            return market_condition
            
        # Автоматическое определение на основе данных
        if 'yield' not in bonds_df.columns:
            return 'neutral'
            
        avg_yield = bonds_df['yield'].mean()
        # Условная логика, которую можно корректировать
        if avg_yield > 12:  # Высокие ставки
            return 'high_rate'
        elif avg_yield < 8:  # Низкие ставки
            return 'low_rate'
        else:
            return 'neutral'
    
    class BondsProcessor:
        def __init__(self, bonds_dir, params):
            self.bonds_dir = bonds_dir
            self.bond_files = self._get_sorted_bond_files()
            self.processed_bonds = None
            self.params = params
            self.market_condition = None
            self.previous_recommendations = None
            self.load_previous_recommendations()
            
        def _get_sorted_bond_files(self):
            """Получение отсортированных файлов с учетом возможных отсутствующих данных"""
            try:
                files = [f for f in os.listdir(self.bonds_dir) if f.endswith('.csv')]
                
                # Сортировка файлов по дате
                def extract_date(filename):
                    date_str = filename.replace('bond_search_', '').replace('.csv', '')
                    if ' ' in date_str:
                        date_str = date_str.split(' ')[0]
                    try:
                        return datetime.strptime(date_str, '%Y-%m-%d')
                    except:
                        try:
                            parts = date_str.split('-')
                            if len(parts) == 3:
                                return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                        except:
                            return datetime(1900, 1, 1)  # Для файлов с неправильным форматом
                
                return sorted(files, key=extract_date)
            except Exception as e:
                logger.error(f"Ошибка при получении списка файлов: {e}")
                return []
                
        def load_previous_recommendations(self):
            """Загрузка предыдущих рекомендаций для учета стабильности портфеля"""
            try:
                prev_file = os.path.join(output_dir, 'latest_recommendations.json')
                if os.path.exists(prev_file):
                    with open(prev_file, 'r') as f:
                        self.previous_recommendations = json.load(f)
                    logger.info(f"Загружены предыдущие рекомендации от {self.previous_recommendations.get('date', 'неизвестной даты')}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить предыдущие рекомендации: {e}")
                self.previous_recommendations = None
            
        def load_and_process_all_bonds(self):
            """Загрузка и обработка всех файлов с облигациями"""
            logger.info(f"Начинаем загрузку и обработку данных по облигациям из {self.bonds_dir}")
            bond_snapshots = []
            
            for file in self.bond_files:
                try:
                    file_path = os.path.join(self.bonds_dir, file)
                    # Извлечение даты из имени файла
                    date_str = file.replace('bond_search_', '').replace('.csv', '')
                    
                    # Обработка случая с временем в имени файла
                    if ' ' in date_str:
                        date_str = date_str.split(' ')[0]
                        
                    # Парсинг даты из имени файла
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        try:
                            parts = date_str.split('-')
                            if len(parts) == 3:
                                formatted_date = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                                date = datetime.strptime(formatted_date, '%Y-%m-%d')
                            else:
                                logger.warning(f"Пропуск файла с неверным форматом даты: {file}")
                                continue
                        except Exception as e:
                            logger.error(f"Ошибка парсинга даты из {file}: {e}")
                            continue
                    
                    # Фильтрация по диапазону дат, если задан
                    if self.params['date_range']:
                        start_date = datetime.strptime(self.params['date_range'][0], '%Y-%m-%d')
                        end_date = datetime.strptime(self.params['date_range'][1], '%Y-%m-%d')
                        if date < start_date or date > end_date:
                            continue
                    
                    # Чтение данных облигаций
                    df = pd.read_csv(file_path)
                    df['snapshot_date'] = date
                    
                    # Проверка, что это данные по облигациям с ожидаемыми колонками
                    expected_columns = ['Полное наименование', 'Код ценной бумаги']
                    if not all(col in df.columns for col in expected_columns):
                        logger.warning(f"Пропуск файла с неожиданным форматом: {file}")
                        continue
                        
                    bond_snapshots.append(df)
                    logger.info(f"Успешно загружен файл {file} с {len(df)} облигациями от {date.strftime('%Y-%m-%d')}")
                except Exception as e:
                    logger.error(f"Ошибка обработки файла {file}: {e}\n{traceback.format_exc()}")
                    continue
            
            if not bond_snapshots:
                logger.error("Не найдены данные по облигациям!")
                return pd.DataFrame()
                
            # Объединение всех снапшотов
            try:
                all_bonds = pd.concat(bond_snapshots, ignore_index=True)
                logger.info(f"Объединены данные по {len(all_bonds)} записям облигаций")
            except Exception as e:
                logger.error(f"Ошибка при объединении данных: {e}")
                return pd.DataFrame()
            
            # Используем только последний снимок если нужно
            if self.params['use_latest_snapshot_only'] and 'snapshot_date' in all_bonds.columns:
                latest_date = all_bonds['snapshot_date'].max()
                all_bonds = all_bonds[all_bonds['snapshot_date'] == latest_date]
                logger.info(f"Применен фильтр: только последний снимок от {latest_date.strftime('%Y-%m-%d')} ({len(all_bonds)} записей)")
            
            # Сопоставление колонок со стандартизированными именами
            self._standardize_columns(all_bonds)
            
            # Очистка и преобразование типов данных
            self._clean_data_types(all_bonds)
            
            # Фильтрация данных согласно параметрам
            filtered_bonds = self._apply_filters(all_bonds)
            
            # Определение рыночных условий
            self.market_condition = detect_market_condition(filtered_bonds)
            logger.info(f"Определено состояние рынка: {self.market_condition}")
            
            # Адаптация весов в зависимости от рыночных условий
            adjusted_weights = self._adjust_weights_for_market_condition()
            self.params['score_weights'] = adjusted_weights
            
            # Расчет ключевых метрик облигаций
            self.processed_bonds = self.calculate_bond_metrics(filtered_bonds)
            logger.info(f"Завершена обработка {len(self.processed_bonds)} облигаций")
            
            return self.processed_bonds
            
        def _standardize_columns(self, df):
            """Стандартизация имен колонок"""
            column_mapping = {
                'Полное наименование': 'full_name',
                'Код ценной бумаги': 'security_code',
                'Цена, %': 'price_pct',
                'Объем сделок за n дней, шт.': 'trading_volume',
                'Доходность': 'yield',
                'Дюрация, месяцев': 'duration_months',
                'Дата погашения': 'expiration_date',
                'Нужна квалификация?': 'qualification_needed',
                'Месяцы выплат': 'payment_months',
                'Есть льгота?': 'tax_benefit',
                'snapshot_date': 'snapshot_date'
            }
            
            # Переименовываем только существующие колонки
            rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df.rename(columns=rename_cols, inplace=True)
            
            # Логирование преобразованных колонок
            logger.info(f"Стандартизированы колонки: {list(rename_cols.values())}")
            
        def _clean_data_types(self, df):
            """Очистка и преобразование типов данных"""
            # Преобразование числовых колонок
            for col in ['price_pct', 'yield', 'duration_months', 'trading_volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Преобразование булевых колонок
            if 'tax_benefit' in df.columns:
                if df['tax_benefit'].dtype == 'object':
                    tax_map = {
                        'да': True, 'true': True, 'True': True, 'TRUE': True, 
                        'нет': False, 'false': False, 'False': False, 'FALSE': False
                    }
                    df['tax_benefit'] = df['tax_benefit'].map(lambda x: tax_map.get(str(x).lower(), False))
            
            # Преобразование дюрации из месяцев в годы
            if 'duration_months' in df.columns:
                df['duration_years'] = df['duration_months'] / 12
                
            # Преобразование дат
            if 'expiration_date' in df.columns:
                df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
                
            logger.info("Завершено преобразование типов данных")
            
        def _apply_filters(self, df):
            """Применение всех фильтров к данным"""
            filtered_df = df.copy()
            initial_count = len(filtered_df)
            
            # Фильтрация по доходности
            if self.params['min_yield'] is not None and 'yield' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['yield'] >= self.params['min_yield']]
                logger.info(f"Фильтр min_yield ({self.params['min_yield']}): осталось {len(filtered_df)} из {initial_count}")
                
            if self.params['max_yield'] is not None and 'yield' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['yield'] <= self.params['max_yield']]
                logger.info(f"Фильтр max_yield ({self.params['max_yield']}): осталось {len(filtered_df)}")
            
            # Фильтрация по дюрации
            if 'duration_years' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['duration_years'] >= self.params['min_duration']) & 
                    (filtered_df['duration_years'] <= self.params['max_duration'])
                ]
                logger.info(f"Фильтр duration ({self.params['min_duration']}-{self.params['max_duration']} лет): осталось {len(filtered_df)}")
            
            # Фильтрация по дате погашения
            if self.params['max_expiration_date'] and 'expiration_date' in filtered_df.columns:
                max_exp = pd.to_datetime(self.params['max_expiration_date'])
                filtered_df = filtered_df[filtered_df['expiration_date'] <= max_exp]
                logger.info(f"Фильтр max_expiration_date ({max_exp.strftime('%Y-%m-%d')}): осталось {len(filtered_df)}")
            
            # Фильтрация по ликвидности
            if self.params['min_liquidity'] is not None and 'trading_volume' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['trading_volume'] >= self.params['min_liquidity']]
                logger.info(f"Фильтр min_liquidity ({self.params['min_liquidity']}): осталось {len(filtered_df)}")
                
            # Исключение конкретных эмитентов
            if self.params['excluded_issuers'] and 'full_name' in filtered_df.columns:
                for issuer in self.params['excluded_issuers']:
                    before = len(filtered_df)
                    filtered_df = filtered_df[~filtered_df['full_name'].str.contains(issuer, case=False, na=False)]
                    logger.info(f"Исключение эмитента '{issuer}': удалено {before - len(filtered_df)} облигаций")
            
            # Исключение конкретных облигаций
            if self.params['excluded_bonds'] and 'security_code' in filtered_df.columns:
                before = len(filtered_df)
                filtered_df = filtered_df[~filtered_df['security_code'].isin(self.params['excluded_bonds'])]
                logger.info(f"Исключение указанных облигаций: удалено {before - len(filtered_df)}")
                
            # Включение только указанных облигаций
            if self.params['include_only'] and 'security_code' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['security_code'].isin(self.params['include_only'])]
                logger.info(f"Включение только указанных облигаций: осталось {len(filtered_df)}")
            
            # Удаление дубликатов по коду ценной бумаги (при необходимости)
            if 'security_code' in filtered_df.columns:
                filtered_df = filtered_df.drop_duplicates(subset=['security_code'], keep='last')
                
            logger.info(f"После применения всех фильтров осталось {len(filtered_df)} облигаций из {initial_count}")
            return filtered_df
        
        def _adjust_weights_for_market_condition(self):
            """Адаптация весов в зависимости от рыночных условий"""
            weights = self.params['score_weights'].copy()
            
            if self.market_condition == 'high_rate':
                # В условиях высоких ставок увеличиваем вес доходности
                logger.info("Корректировка весов для условий высоких ставок")
                weights['yield'] = min(0.7, weights['yield'] * 1.3)
                weights['duration'] = max(0.1, weights['duration'] * 0.7)
                
            elif self.market_condition == 'low_rate':
                # В условиях низких ставок увеличиваем вес дюрации
                logger.info("Корректировка весов для условий низких ставок")
                weights['yield'] = max(0.3, weights['yield'] * 0.8)
                weights['duration'] = min(0.5, weights['duration'] * 1.5)
                
            # Нормализация весов до суммы 1
            total = sum(weights.values())
            if total != 1.0:
                weights = {k: v / total for k, v in weights.items()}
                
            logger.info(f"Адаптированные веса: {weights}")
            return weights
            
        def calculate_bond_metrics(self, bonds_df):
            """Расчет дополнительных метрик с настраиваемыми весами"""
            df = bonds_df.copy()
            
            try:
                # Расчет доходности с поправкой на риск (доходность / дюрация)
                if 'yield' in df.columns and 'duration_years' in df.columns:
                    df['risk_adjusted_yield'] = df['yield'] / df['duration_years'].clip(lower=0.1)
                
                # Расчет z-оценок для доходности внутри каждого снапшота
                if 'yield' in df.columns and 'snapshot_date' in df.columns:
                    df['yield_zscore'] = df.groupby('snapshot_date')['yield'].transform(
                        lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
                    )
                
                # Расчет отклонения цены от номинала (100%)
                if 'price_pct' in df.columns:
                    df['price_deviation'] = df['price_pct'] - 100
                
                # Расчет скора облигации с настраиваемыми весами
                score_components = []
                
                if 'yield_zscore' in df.columns and 'yield' in self.params['score_weights']:
                    score_components.append(df['yield_zscore'] * self.params['score_weights']['yield'])
                    
                if 'duration_years' in df.columns and 'duration' in self.params['score_weights']:
                    score_components.append(
                        (1 / df['duration_years'].clip(0.5, None)) * self.params['score_weights']['duration']
                    )
                    
                if 'tax_benefit' in df.columns and 'tax_benefit' in self.params['score_weights']:
                    score_components.append(
                        df['tax_benefit'].astype(float) * self.params['score_weights']['tax_benefit']
                    )
                                
                if 'price_deviation' in df.columns and 'price_discount' in self.params['score_weights']:
                    score_components.append(
                        (df['price_deviation'] < -1).astype(float) * self.params['score_weights']['price_discount']
                    )
                
                if score_components:
                    # Правильное суммирование компонентов
                    df['bond_score'] = pd.concat(score_components, axis=1).sum(axis=1)
                else:
                    df['bond_score'] = 0.5  # Нейтральный скор по умолчанию
                
                logger.info(f"Рассчитаны метрики для {len(df)} облигаций")
                return df
                
            except Exception as e:
                logger.error(f"Ошибка при расчете метрик: {e}\n{traceback.format_exc()}")
                # Возвращаем исходные данные при ошибке
                return bonds_df
        
        def get_bonds_for_portfolio(self, n_bonds=5):
            """Получение рекомендованных облигаций с учетом стабильности портфеля"""
            if self.processed_bonds is None or len(self.processed_bonds) == 0:
                logger.warning("Нет доступных данных по облигациям для формирования портфеля")
                return pd.DataFrame()
                        
            try:
                # Получаем последние данные
                if 'snapshot_date' in self.processed_bonds.columns and self.params['use_latest_snapshot_only']:
                    latest_date = self.processed_bonds['snapshot_date'].max()
                    bond_pool = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
                else:
                    bond_pool = self.processed_bonds.copy()
                
                if len(bond_pool) == 0:
                    logger.warning("Нет подходящих облигаций для портфеля")
                    return pd.DataFrame()
                
                logger.info(f"Доступно {len(bond_pool)} облигаций для формирования портфеля")
                logger.info(f"Колонки для отбора: {list(bond_pool.columns)}")
                
                # Выберем больше облигаций, чем нужно (для учета стабильности)
                expanded_n = min(int(n_bonds * 1.5) + 2, len(bond_pool))
                
                # Выбор лучших облигаций по скору
                if 'bond_score' in bond_pool.columns:
                    top_candidates = bond_pool.nlargest(expanded_n, 'bond_score')
                else:
                    top_candidates = bond_pool.head(expanded_n)
                
                # Применение логики стабильности портфеля
                top_bonds = self._apply_portfolio_stability(top_candidates, n_bonds)
                
                # Создаем сводку для включения в портфель
                columns_to_keep = ['security_code', 'full_name', 'yield', 'duration_years', 
                                  'risk_adjusted_yield', 'bond_score']
                columns_to_keep = [col for col in columns_to_keep if col in top_bonds.columns]
                
                bond_summary = top_bonds[columns_to_keep].copy()
                
                # Расчет весов на основе выбранной стратегии
                bond_summary = self._calculate_weights(bond_summary)
                
                # Добавление информации о рыночных условиях
                bond_summary.attrs['market_condition'] = self.market_condition
                bond_summary.attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Сохранение текущих рекомендаций для будущего сравнения
                self._save_current_recommendations(bond_summary)
                
                logger.info(f"Сформирован портфель из {len(bond_summary)} облигаций")
                return bond_summary
                
            except Exception as e:
                logger.error(f"Ошибка при формировании портфеля: {e}\n{traceback.format_exc()}")
                return pd.DataFrame()
                
        def _apply_portfolio_stability(self, candidates, n_bonds):
            """Применение логики стабильности портфеля для минимизации ребалансировок"""
            stability = self.params.get('portfolio_stability', 0.7)
            
            if not self.previous_recommendations or 'bonds' not in self.previous_recommendations:
                logger.info("Нет предыдущих рекомендаций - используем только текущие данные")
                return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
            
            try:
                # Получаем предыдущие рекомендации
                prev_bonds = self.previous_recommendations['bonds']
                prev_codes = [b['security_code'] for b in prev_bonds if 'security_code' in b]
                
                # Определяем количество облигаций, которые нужно сохранить
                keep_n = int(n_bonds * stability)
                
                # Находим облигации, которые были в предыдущем портфеле и есть в текущих кандидатах
                kept_bonds = candidates[candidates['security_code'].isin(prev_codes)]
                
                if len(kept_bonds) > 0:
                    # Отсортируем сохраняемые облигации по скору
                    kept_bonds = kept_bonds.nlargest(min(keep_n, len(kept_bonds)), 'bond_score') if 'bond_score' in kept_bonds.columns else kept_bonds.head(min(keep_n, len(kept_bonds)))
                    
                    # Выберем новые облигации, исключив те, что уже сохранили
                    new_n = n_bonds - len(kept_bonds)
                    new_bonds = candidates[~candidates['security_code'].isin(kept_bonds['security_code'])].nlargest(new_n, 'bond_score') if 'bond_score' in candidates.columns else candidates[~candidates['security_code'].isin(kept_bonds['security_code'])].head(new_n)
                    
                    # Объединяем сохраненные и новые облигации
                    result = pd.concat([kept_bonds, new_bonds])
                    logger.info(f"Сохранено {len(kept_bonds)} облигаций из предыдущего портфеля, добавлено {len(new_bonds)} новых")
                    
                    return result
                else:
                    logger.info("Нет пересечений с предыдущим портфелем - используем только новые данные")
                    return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
                    
            except Exception as e:
                logger.warning(f"Ошибка при применении стабильности портфеля: {e} - используем только текущие данные")
                return candidates.nlargest(n_bonds, 'bond_score') if 'bond_score' in candidates.columns else candidates.head(n_bonds)
                
        def _calculate_weights(self, bond_summary):
            """Расчет весов на основе выбранной стратегии"""
            weighting = self.params['weighting_strategy']
            
            try:
                if weighting == 'equal':
                    # Равные веса
                    bond_summary['weight'] = 1.0 / len(bond_summary)
                    
                elif weighting == 'inverse_duration' and 'duration_years' in bond_summary.columns:
                    # Веса обратно пропорциональны дюрации
                    inverse_duration = 1 / bond_summary['duration_years'].clip(lower=0.1)
                    bond_summary['weight'] = inverse_duration / inverse_duration.sum()
                    
                elif weighting == 'yield' and 'yield' in bond_summary.columns:
                    # Веса пропорциональны доходности
                    bond_summary['weight'] = bond_summary['yield'] / bond_summary['yield'].sum()
                    
                elif weighting == 'bond_score' and 'bond_score' in bond_summary.columns:
                    # Веса пропорциональны скору облигации
                    scores = bond_summary['bond_score'].clip(lower=0.01)  # Избегаем отрицательных значений
                    bond_summary['weight'] = scores / scores.sum()
                    
                else:
                    # Равные веса по умолчанию
                    bond_summary['weight'] = 1.0 / len(bond_summary)
                
                # Округляем веса для читаемости
                bond_summary['weight'] = bond_summary['weight'].round(4)
                logger.info(f"Рассчитаны веса по стратегии '{weighting}'")
                
                return bond_summary
            
            except Exception as e:
                logger.error(f"Ошибка при расчете весов: {e} - используем равные веса")
                bond_summary['weight'] = 1.0 / len(bond_summary)
                return bond_summary
                
        def _save_current_recommendations(self, bond_summary):
            """Сохранение текущих рекомендаций для будущего сравнения"""
            if self.params['save_results']:
                try:
                    recommendations = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'market_condition': self.market_condition,
                        'bonds': bond_summary.to_dict(orient='records')
                    }
                    
                    # Сохраняем в JSON для будущего использования
                    with open(os.path.join(output_dir, 'latest_recommendations.json'), 'w') as f:
                        json.dump(recommendations, f, indent=2)
                        
                    logger.info("Текущие рекомендации сохранены для будущего использования")
                except Exception as e:
                    logger.error(f"Ошибка при сохранении рекомендаций: {e}")
        
        def visualize_bond_universe(self):
            """Создание визуализаций с учетом рыночных условий"""
            if self.processed_bonds is None or len(self.processed_bonds) == 0:
                logger.warning("Нет данных для визуализации")
                return None
            
            try:
                # Создаем директорию для визуализаций
                vis_dir = os.path.join(output_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Получаем данные для визуализации
                if self.params['use_latest_snapshot_only'] and 'snapshot_date' in self.processed_bonds.columns:
                    latest_date = self.processed_bonds['snapshot_date'].max()
                    plot_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date].copy()
                    date_title = f"{latest_date.strftime('%Y-%m-%d')}"
                else:
                    plot_bonds = self.processed_bonds.copy()
                    if self.params['date_range']:
                        date_title = f"{self.params['date_range'][0]} до {self.params['date_range'][1]}"
                    else:
                        date_title = "Все доступные данные"
                
                # Создаем график доходность/дюрация
                if 'duration_years' in plot_bonds.columns and 'yield' in plot_bonds.columns:
                    plt.figure(figsize=(12, 8))
                    
                    # Основной график
                    scatter = plt.scatter(
                        plot_bonds['duration_years'], 
                        plot_bonds['yield'], 
                        alpha=0.7, s=100, 
                        c=plot_bonds['bond_score'] if 'bond_score' in plot_bonds.columns else None, 
                        cmap='viridis'
                    )
                    
                    # Аннотация для топ-облигаций
                    if 'bond_score' in plot_bonds.columns:
                        top_bonds = plot_bonds.nlargest(5, 'bond_score')
                        for _, bond in top_bonds.iterrows():
                            plt.annotate(
                                bond['security_code'],
                                (bond['duration_years'], bond['yield']),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=9,
                                fontweight='bold'
                            )
                    
                    if 'bond_score' in plot_bonds.columns:
                        plt.colorbar(scatter, label='Скор облигации')
                    
                    market_info = f" | Состояние рынка: {self.market_condition}" if self.market_condition else ""
                    plt.title(f'Облигации: Доходность vs Дюрация ({date_title}){market_info}')
                    plt.xlabel('Дюрация (лет)')
                    plt.ylabel('Доходность (%)')
                    plt.grid(True, alpha=0.3)
                    
                    # Сохраняем визуализацию
                    today = datetime.now().strftime('%Y%m%d')
                    plt.savefig(f'{vis_dir}/bond_yield_curve_{today}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Создаем график риск/доходность
                    if 'risk_adjusted_yield' in plot_bonds.columns:
                        plt.figure(figsize=(12, 8))
                        
                        plt.scatter(
                            plot_bonds['yield'], 
                            plot_bonds['risk_adjusted_yield'], 
                            alpha=0.7, s=100, 
                            c=plot_bonds['duration_years'], 
                            cmap='coolwarm'
                        )
                        
                        # Аннотация для топ-облигаций по скорректированной на риск доходности
                        top_raj_bonds = plot_bonds.nlargest(5, 'risk_adjusted_yield')
                        for _, bond in top_raj_bonds.iterrows():
                            plt.annotate(
                                bond['security_code'],
                                (bond['yield'], bond['risk_adjusted_yield']),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=9,
                                fontweight='bold'
                            )
                        
                        plt.colorbar(label='Дюрация (лет)')
                        plt.title(f'Профиль риск-доходность облигаций ({date_title}){market_info}')
                        plt.xlabel('Доходность (%)')
                        plt.ylabel('Скорректированная на риск доходность (Доходность/Дюрация)')
                        plt.grid(True, alpha=0.3)
                        
                        # Сохраняем визуализацию
                        plt.savefig(f'{vis_dir}/bond_risk_return_{today}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                
                # Создаем визуализацию для сравнения с предыдущими рекомендациями
                if self.params['comparison_with_previous'] and self.previous_recommendations:
                    self._visualize_comparison_with_previous()
                
                logger.info("Создание визуализаций завершено")
                return plot_bonds
                
            except Exception as e:
                logger.error(f"Ошибка при создании визуализаций: {e}\n{traceback.format_exc()}")
                return None
                
        def _visualize_comparison_with_previous(self):
            """Визуализация сравнения с предыдущими рекомендациями"""
            try:
                # Получаем текущие топ-облигации
                if self.processed_bonds is None or len(self.processed_bonds) == 0:
                    return
                    
                top_bonds = self.get_bonds_for_portfolio(n_bonds=self.params['select_top_n'])
                if len(top_bonds) == 0:
                    return
                    
                # Получаем предыдущие рекомендации
                if not self.previous_recommendations or 'bonds' not in self.previous_recommendations:
                    return
                    
                prev_bonds = pd.DataFrame(self.previous_recommendations['bonds'])
                prev_date = self.previous_recommendations.get('date', 'прошлый раз')
                
                # Создаем сравнение изменений в весах
                if 'security_code' in prev_bonds.columns and 'weight' in prev_bonds.columns:
                    comparison = []
                    
                    # Облигации, которые есть в обоих портфелях
                    for code in top_bonds['security_code']:
                        if code in prev_bonds['security_code'].values:
                            prev_weight = prev_bonds.loc[prev_bonds['security_code'] == code, 'weight'].values[0]
                            curr_weight = top_bonds.loc[top_bonds['security_code'] == code, 'weight'].values[0]
                            comparison.append({
                                'security_code': code,
                                'previous_weight': prev_weight,
                                'current_weight': curr_weight,
                                'weight_change': curr_weight - prev_weight,
                                'status': 'сохранен'
                            })
                    
                    # Новые облигации, которых не было раньше
                    for code in top_bonds['security_code']:
                        if code not in prev_bonds['security_code'].values:
                            curr_weight = top_bonds.loc[top_bonds['security_code'] == code, 'weight'].values[0]
                            comparison.append({
                                'security_code': code,
                                'previous_weight': 0,
                                'current_weight': curr_weight,
                                'weight_change': curr_weight,
                                'status': 'новый'
                            })
                    
                    # Удаленные облигации, которых больше нет
                    for code in prev_bonds['security_code']:
                        if code not in top_bonds['security_code'].values:
                            prev_weight = prev_bonds.loc[prev_bonds['security_code'] == code, 'weight'].values[0]
                            comparison.append({
                                'security_code': code,
                                'previous_weight': prev_weight,
                                'current_weight': 0,
                                'weight_change': -prev_weight,
                                'status': 'удален'
                            })
                    
                    # Создаем DataFrame для сравнения
                    comparison_df = pd.DataFrame(comparison)
                    
                    # Создаем график изменений весов
                    plt.figure(figsize=(12, 8))
                    
                    # Цвета для различных статусов
                    colors = {'сохранен': 'blue', 'новый': 'green', 'удален': 'red'}
                    
                    # Сортируем по изменению веса
                    comparison_df = comparison_df.sort_values('weight_change')
                    
                    # Создаем бар-график изменений
                    bars = plt.barh(
                        comparison_df['security_code'], 
                        comparison_df['weight_change'],
                        color=[colors[status] for status in comparison_df['status']]
                    )
                    
                    # Добавляем метки с текущими и предыдущими весами
                    for i, bar in enumerate(bars):
                        prev = comparison_df.iloc[i]['previous_weight']
                        curr = comparison_df.iloc[i]['current_weight']
                        plt.text(
                            0, bar.get_y() + bar.get_height()/2,
                            f"{prev:.2%} → {curr:.2%}",
                            va='center', fontsize=9
                        )
                    
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.title(f'Изменения в портфеле облигаций по сравнению с {prev_date}')
                    plt.xlabel('Изменение веса')
                    plt.ylabel('Код облигации')
                    plt.grid(True, alpha=0.3, axis='x')
                    
                    # Создаем легенду
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', label='Новые'),
                        Patch(facecolor='blue', label='Сохраненные'),
                        Patch(facecolor='red', label='Удаленные')
                    ]
                    plt.legend(handles=legend_elements)
                    
                    # Сохраняем визуализацию
                    vis_dir = os.path.join(output_dir, 'visualizations')
                    today = datetime.now().strftime('%Y%m%d')
                    plt.savefig(f'{vis_dir}/bond_portfolio_changes_{today}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("Создана визуализация изменений в портфеле")
            
            except Exception as e:
                logger.error(f"Ошибка при визуализации сравнения с предыдущими рекомендациями: {e}")
        
        def get_risk_free_rate(self):
            """Получение безрисковой ставки с учетом рыночных условий"""
            if self.processed_bonds is None or len(self.processed_bonds) == 0:
                logger.warning("Нет данных для расчета безрисковой ставки, используем значение по умолчанию")
                return 0.075  # По умолчанию
                
            try:
                # Получаем последний снимок
                if 'snapshot_date' in self.processed_bonds.columns:
                    latest_date = self.processed_bonds['snapshot_date'].max()
                    latest_bonds = self.processed_bonds[self.processed_bonds['snapshot_date'] == latest_date]
                else:
                    latest_bonds = self.processed_bonds
                
                if 'yield' not in latest_bonds.columns or 'duration_years' not in latest_bonds.columns:
                    logger.warning("Отсутствуют необходимые колонки для расчета безрисковой ставки")
                    return 0.075  # По умолчанию
                
                # Находим краткосрочные качественные облигации
                short_bonds = latest_bonds[latest_bonds['duration_years'] <= 1.5]
                
                if len(short_bonds) > 0:
                    # Используем среднее значение для топ-3 облигаций с минимальной дюрацией
                    risk_free = short_bonds.nsmallest(3, 'duration_years')['yield'].mean() / 100
                    logger.info(f"Рассчитана безрисковая ставка: {risk_free:.2%} на основе {len(short_bonds)} облигаций")
                    return risk_free
                
                # Запасной вариант - используем среднюю доходность всех облигаций с минимальной дюрацией
                if len(latest_bonds) > 0:
                    risk_free = latest_bonds.nsmallest(3, 'duration_years')['yield'].mean() / 100
                    logger.info(f"Рассчитана безрисковая ставка (запасной вариант): {risk_free:.2%}")
                    return risk_free
                
                # Значение по умолчанию
                logger.warning("Не удалось рассчитать безрисковую ставку, используем значение по умолчанию")
                return 0.075
                
            except Exception as e:
                logger.error(f"Ошибка при расчете безрисковой ставки: {e}")
                return 0.075
                
        def export_results(self, top_bonds, risk_free_rate):
            """Экспорт результатов в выбранный формат"""
            if top_bonds.empty:
                logger.warning("Нет данных для экспорта")
                return
                
            export_format = self.params.get('export_format', 'excel')
            today = datetime.now().strftime('%Y%m%d')
            
            try:
                # Добавляем дополнительную информацию
                export_bonds = top_bonds.copy()
                
                # Создаем полную информацию для экспорта
                export_data = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'market_condition': self.market_condition,
                    'risk_free_rate': f"{risk_free_rate:.2%}",
                    'pipeline_parameters': {k: str(v) if isinstance(v, dict) else v for k, v in self.params.items()},
                    'bonds': export_bonds.to_dict(orient='records')
                }
                
                # Экспорт в выбранный формат
                if export_format == 'excel':
                    export_file = os.path.join(output_dir, f'bond_portfolio_{today}.xlsx')
                    with pd.ExcelWriter(export_file) as writer:
                        export_bonds.to_excel(writer, sheet_name='Рекомендации', index=False)
                        
                        # Создаем лист с метаданными
                        meta_df = pd.DataFrame([
                            ['Дата', export_data['date']],
                            ['Состояние рынка', export_data['market_condition']],
                            ['Безрисковая ставка', export_data['risk_free_rate']],
                            ['Параметры весов', str(self.params['score_weights'])],
                            ['Стратегия взвешивания', self.params['weighting_strategy']],
                        ], columns=['Параметр', 'Значение'])
                        meta_df.to_excel(writer, sheet_name='Метаданные', index=False)
                        
                elif export_format == 'csv':
                    export_file = os.path.join(output_dir, f'bond_portfolio_{today}.csv')
                    export_bonds.to_csv(export_file, index=False)
                    
                    # Сохраняем метаданные в отдельный файл
                    meta_file = os.path.join(output_dir, f'bond_portfolio_meta_{today}.csv')
                    meta_df = pd.DataFrame([
                        ['date', export_data['date']],
                        ['market_condition', export_data['market_condition']],
                        ['risk_free_rate', export_data['risk_free_rate']]
                    ], columns=['key', 'value'])
                    meta_df.to_csv(meta_file, index=False)
                    
                elif export_format == 'json':
                    export_file = os.path.join(output_dir, f'bond_portfolio_{today}.json')
                    with open(export_file, 'w') as f:
                        json.dump(export_data, f, indent=2)
                
                logger.info(f"Результаты экспортированы в {export_file}")
                return export_file
                
            except Exception as e:
                logger.error(f"Ошибка при экспорте результатов: {e}")
                return None

    # Подготовка параметров
    params = {
        'date_range': date_range,
        'use_latest_snapshot_only': use_latest_snapshot_only,
        'min_yield': min_yield,
        'max_yield': max_yield,
        'min_duration': min_duration,
        'max_duration': max_duration,
        'max_expiration_date': max_expiration_date,
        'min_liquidity': min_liquidity,
        'score_weights': score_weights,
        'weighting_strategy': weighting_strategy,
        'excluded_issuers': excluded_issuers or [],
        'excluded_bonds': excluded_bonds or [],
        'include_only': include_only,
        'portfolio_stability': portfolio_stability,
        'export_format': export_format,
        'select_top_n': select_top_n,
        'comparison_with_previous': comparison_with_previous,
        'save_results': save_results,
        'detailed_logging': detailed_logging,
        'market_condition': market_condition
    }

    # Создание процессора и запуск пайплайна
    try:
        processor = BondsProcessor(bonds_dir=bonds_dir, params=params)
        processed_bonds = processor.load_and_process_all_bonds()
        
        if len(processed_bonds) == 0:
            logger.error("Не удалось загрузить и обработать данные по облигациям")
            return {
                'processed_bonds': pd.DataFrame(),
                'risk_free_rate': 0.075,
                'top_bonds': pd.DataFrame(),
                'processor': processor,
                'error': "Не удалось загрузить и обработать данные по облигациям"
            }
        
        if visualize:
            processor.visualize_bond_universe()
            
        risk_free_rate = processor.get_risk_free_rate()
        top_bonds = processor.get_bonds_for_portfolio(n_bonds=select_top_n)

        print("\nТоп рекомендуемых облигаций:")
        if not top_bonds.empty:
            print(top_bonds)
        else:
            print("Не удалось получить рекомендации по облигациям")
        
        # Экспорт результатов в файл если нужно
        export_file = None
        if save_results and len(top_bonds) > 0:
            export_file = processor.export_results(top_bonds, risk_free_rate)
        
        logger.info(f"Пайплайн успешно завершен. Обработано {len(processed_bonds)} облигаций, отобрано {len(top_bonds)}")
        
        # Данные для консольного вывода
        print(f"Обработано {len(processed_bonds)} записей облигаций")
        print(f"Текущая безрисковая ставка: {risk_free_rate*100:.2f}%")
        
        if len(top_bonds) > 0:
            print("\nТоп рекомендуемых облигаций:")
            print(top_bonds)
        
        # Возвращаем результаты выполнения пайплайна
        return {
            'processed_bonds': processed_bonds,
            'risk_free_rate': risk_free_rate,
            'top_bonds': top_bonds,
            'processor': processor,
            'export_file': export_file,
            'market_condition': processor.market_condition
        }
        
    except Exception as e:
        error_msg = f"Критическая ошибка при выполнении пайплайна: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"ОШИБКА: {error_msg}")
        return {
            'processed_bonds': pd.DataFrame(),
            'risk_free_rate': 0.075,
            'top_bonds': pd.DataFrame(),
            'error': error_msg
        }
