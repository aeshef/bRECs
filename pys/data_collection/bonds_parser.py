from datetime import datetime, timedelta
import os
import json
import time
import pandas as pd
import requests
import shutil
from typing import Dict, List, Optional, Any, Tuple, Union
import sys

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH

class MOEXBondHistoricalParser(BaseLogger):
    """
    Парсер для получения исторических данных по облигациям Московской биржи
    с возможностью применения различных фильтров.
    """
    
    BOARD_GROUPS = [58, 193, 105, 77, 207, 167, 245]
    
    API_DELAY = 1.2
    
    CACHE_DIR = f"{BASE_PATH}/data/processed_data/BONDS/moex/moex_cache"
    BASE_DATA_DIR = f"{BASE_PATH}/data/processed_data/BONDS/moex"
    
    def __init__(self, use_cache=True, cache_ttl_days=7, verbose=False):
        """
        Инициализация парсера
        
        Args:
            use_cache: использовать ли кэширование запросов
            cache_ttl_days: срок хранения кэша в днях
            verbose: выводить ли подробные логи
        """
        super().__init__('MOEXBondHistoricalParser')
        self.use_cache = use_cache
        self.cache_ttl_days = cache_ttl_days
        self.verbose = verbose
        
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            
        if not os.path.exists(self.BASE_DATA_DIR):
            os.makedirs(self.BASE_DATA_DIR, exist_ok=True)
            
        self.logger.info("MOEX Bond Parser initialized")
    
    def get_latest_bonds(self, **filter_params):
        """
        Получить актуальные облигации на последний торговый день
        
        Args:
            **filter_params: параметры фильтрации (см. get_available_bonds)
            
        Returns:
            Список облигаций
        """
        last_trading_day = self._get_last_trading_day()
        self.logger.info(f"Получение облигаций на последний торговый день: {last_trading_day.strftime('%Y-%m-%d')}")
        
        return self.get_available_bonds(date=last_trading_day, **filter_params)
    
    def get_available_bonds(self, date=None, 
                        yield_range=(0, 100),
                        price_range=(0, 200), 
                        duration_range=(0, 360),
                        volume_threshold=2000,
                        bond_volume_threshold=60000,
                        is_qualified_investors=None,
                        coupon_type=None,
                        emitent=None,
                        issue_year_range=None,
                        known_coupon_payments=True,
                        use_existing_data=True,
                        load_all_csv=True):
        """
        Получить доступные облигации на указанную дату с применением фильтров
        
        Args:
            date: дата, на которую нужно получить список облигаций
            yield_range: диапазон доходности (%)
            price_range: диапазон цены (%)
            duration_range: диапазон дюрации (месяцы)
            volume_threshold: минимальный объем сделок в день (шт)
            bond_volume_threshold: минимальный совокупный объем сделок (шт)
            is_qualified_investors: требуется ли квалификация инвестора
            coupon_type: тип купона (фиксированный, плавающий и т.д.)
            emitent: эмитент облигации
            issue_year_range: диапазон годов выпуска
            known_coupon_payments: требуется ли наличие всех известных купонных выплат
            use_existing_data: использовать существующие CSV-данные если доступны
            load_all_csv: загружать ли все облигации в all.csv
            
        Returns:
            Список словарей с информацией об облигациях
        """
        if date is None:
            date = self._get_last_trading_day()
        
        date_str = date.strftime('%Y-%m-%d')
        self.logger.info(f"Получение списка облигаций на {date_str}")
        
        date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        all_bonds_path = os.path.join(date_dir, "all.csv")
        all_bonds = None

        if use_existing_data and os.path.exists(all_bonds_path):
            try:
                self.logger.info(f"Используем существующий файл данных: {all_bonds_path}")
                all_bonds_df = pd.read_csv(all_bonds_path)
                all_bonds = all_bonds_df.to_dict('records')
            except Exception as e:
                self.logger.error(f"Ошибка при чтении существующего файла: {e}")
                all_bonds = None
        
        if all_bonds is None:
            all_bonds = self._get_all_bonds_for_date(date)
            if all_bonds and load_all_csv:
                all_bonds_df = pd.DataFrame(all_bonds)
                all_bonds_df.to_csv(all_bonds_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"Сохранены все облигации ({len(all_bonds)}) в {all_bonds_path}")
        
        next_load_number = self._get_next_load_number(date_dir)
        filter_dir = os.path.join(date_dir, f"load_{next_load_number}")
        os.makedirs(filter_dir, exist_ok=True)
        
        filtered_bonds = self._apply_filters(
            all_bonds, 
            date,
            yield_range, 
            price_range, 
            duration_range,
            volume_threshold,
            bond_volume_threshold,
            is_qualified_investors,
            coupon_type,
            emitent,
            issue_year_range,
            known_coupon_payments
        )
        
        filtered_path = os.path.join(filter_dir, "filtered.csv")
        if filtered_bonds:
            filtered_df = pd.DataFrame(filtered_bonds)
            filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Сохранены отфильтрованные облигации ({len(filtered_bonds)}) в {filtered_path}")
        
        filter_params = {
            "yield_range": yield_range,
            "price_range": price_range,
            "duration_range": duration_range,
            "volume_threshold": volume_threshold,
            "bond_volume_threshold": bond_volume_threshold,
            "is_qualified_investors": is_qualified_investors,
            "coupon_type": coupon_type,
            "emitent": emitent,
            "issue_year_range": issue_year_range,
            "known_coupon_payments": known_coupon_payments,
            "date": date_str,
            "total_bonds": len(all_bonds),
            "filtered_bonds": len(filtered_bonds),
            "load_number": next_load_number
        }
        
        with open(os.path.join(filter_dir, "params.json"), 'w', encoding='utf-8') as f:
            json.dump(filter_params, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Найдено {len(filtered_bonds)} облигаций после фильтрации (загрузка #{next_load_number})")
        
        return filtered_bonds
    
    def _duplicate_trading_day_data(self, source_date, target_date):
        """
        Дублировать данные торгового дня для неторгового дня
        
        Args:
            source_date: дата-источник (торговый день)
            target_date: целевая дата (неторговый день)
        """
        source_dir = os.path.join(self.BASE_DATA_DIR, source_date.strftime('%Y-%m-%d'))
        target_dir = os.path.join(self.BASE_DATA_DIR, target_date.strftime('%Y-%m-%d'))
        
        if not os.path.exists(source_dir):
            self.logger.warning(f"Исходная директория {source_dir} не существует")
            return
        
        os.makedirs(target_dir, exist_ok=True)
        
        source_all_csv = os.path.join(source_dir, "all.csv")
        target_all_csv = os.path.join(target_dir, "all.csv")
        
        if os.path.exists(source_all_csv):
            try:
                df = pd.read_csv(source_all_csv)
                
                df.to_csv(target_all_csv, index=False, encoding='utf-8-sig')
                self.logger.info(f"Скопированы данные из {source_all_csv} в {target_all_csv}")
            except Exception as e:
                self.logger.error(f"Ошибка при копировании данных: {e}")
        
    def get_bonds_for_period(self, start_date, end_date, include_non_trading_days=True, load_all_csv=True, **filter_params):
        """
        Получить облигации за указанный период с применением фильтров
        
        Args:
            start_date: начало периода
            end_date: конец периода
            include_non_trading_days: включать ли нерабочие дни (выходные и праздники)
            **filter_params: параметры фильтрации
            
        Returns:
            Словарь с датами и списками облигаций
        """
        last_trading_day = self._get_last_trading_day()
        
        self.logger.info(f"Запрошен период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")

        if start_date > last_trading_day:
            start_date = last_trading_day
            self.logger.warning(f"Скорректирована начальная дата на {start_date.strftime('%Y-%m-%d')}")
            
        if end_date > last_trading_day:
            end_date = last_trading_day
            self.logger.warning(f"Скорректирована конечная дата на {end_date.strftime('%Y-%m-%d')}")
        
        if end_date < start_date:
            start_date, end_date = end_date, start_date
            self.logger.warning("Даты начала и конца периода были поменяны местами")
        
        self.logger.info(f"Получение облигаций за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        period_dir_name = f"period_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        period_dir = os.path.join(self.BASE_DATA_DIR, period_dir_name)
        os.makedirs(period_dir, exist_ok=True)
        
        filter_folder_name = self._create_filter_folder_name(**filter_params)
        filter_dir = os.path.join(period_dir, filter_folder_name)
        os.makedirs(filter_dir, exist_ok=True)
        
        result = {}
        current_date = start_date
        
        last_trading_data = None
        last_trading_date = None
        
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            is_trading_day = current_date.weekday() < 5  # 0-4: понедельник-пятница
            
            self.logger.info(f"Обработка даты: {date_key} ({'рабочий день' if is_trading_day else 'выходной'})")
            
            if is_trading_day:
                bonds = self.get_available_bonds(
                    date=current_date,
                    load_all_csv=load_all_csv,
                    **filter_params
                )
                
                if bonds:
                    result[date_key] = bonds
                    self.logger.info(f"На {date_key} найдено {len(bonds)} облигаций")
                    
                    last_trading_data = bonds
                    last_trading_date = current_date
                else:
                    self.logger.warning(f"На {date_key} облигаций не найдено, возможно праздник или выходной")
                    
                    if include_non_trading_days and last_trading_data:
                        result[date_key] = last_trading_data
                        self.logger.info(f"Использованы данные последнего торгового дня ({last_trading_date.strftime('%Y-%m-%d')}) для {date_key}")
                        self._duplicate_trading_day_data(last_trading_date, current_date)
                        
            elif include_non_trading_days and last_trading_data:
                result[date_key] = last_trading_data
                self.logger.info(f"Использованы данные последнего торгового дня ({last_trading_date.strftime('%Y-%m-%d')}) для {date_key}")
                
                self._duplicate_trading_day_data(last_trading_date, current_date)
                
            current_date += timedelta(days=1)
        
        all_bonds = []
        for date, bonds in result.items():
            for bond in bonds:
                bond_with_date = bond.copy()
                bond_with_date['date'] = date
                all_bonds.append(bond_with_date)
        
        if all_bonds:
            period_df = pd.DataFrame(all_bonds)
            period_file = os.path.join(filter_dir, "period_data.csv")
            period_df.to_csv(period_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"Сохранены результаты за период в {period_file}")
            
            summary = {}
            for date, bonds in result.items():
                summary[date] = len(bonds)
                
            summary_df = pd.DataFrame(list(summary.items()), columns=['date', 'bond_count'])
            summary_df.to_csv(os.path.join(filter_dir, "period_summary.csv"), index=False)
        
        filter_params['period_start'] = start_date.strftime('%Y-%m-%d')
        filter_params['period_end'] = end_date.strftime('%Y-%m-%d')
        filter_params['total_dates'] = len(result)
        filter_params['total_bonds'] = sum(len(bonds) for bonds in result.values())
        filter_params['include_non_trading_days'] = include_non_trading_days
        
        with open(os.path.join(filter_dir, "params.json"), 'w', encoding='utf-8') as f:
            json.dump(filter_params, f, ensure_ascii=False, indent=2)
        
        return result
    
    def _create_filter_folder_name(self, yield_range=(0, 100), price_range=(0, 200), 
                                  duration_range=(0, 360), volume_threshold=2000,
                                  bond_volume_threshold=60000, is_qualified_investors=None,
                                  coupon_type=None, emitent=None, issue_year_range=None,
                                  known_coupon_payments=True, **kwargs):
        """Создать имя папки на основе параметров фильтрации"""
        parts = []
        
        parts.append(f"yieldrange_{yield_range[0]}_{yield_range[1]}")
        parts.append(f"pricerange_{price_range[0]}_{price_range[1]}")
        parts.append(f"durationrange_{duration_range[0]}_{duration_range[1]}")
    
        parts.append(f"volthresh_{volume_threshold}")
        parts.append(f"bondvolthresh_{bond_volume_threshold}")
        
        if is_qualified_investors is not None:
            parts.append(f"qualified_{str(is_qualified_investors).lower()}")
            
        if coupon_type is not None:
            parts.append(f"coupontype_{coupon_type}")
            
        if emitent is not None:
            safe_emitent = "".join(c if c.isalnum() else "_" for c in emitent.lower())
            parts.append(f"emitent_{safe_emitent}")
            
        if issue_year_range is not None:
            parts.append(f"issueyear_{issue_year_range[0]}_{issue_year_range[1]}")
            
        if known_coupon_payments:
            parts.append("known_coupons")
        
        folder_name = "__".join(parts)
        
        if len(folder_name) > 200:
            folder_name = folder_name[:200]
            
        return folder_name
    
    def _get_all_bonds_for_date(self, date):
        """
        Получить все доступные облигации на указанную дату
        """
        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"all_bonds_{date_str}"
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            self.logger.info(f"Используем кэшированные данные для {date_str}")
            return cached_data
        
        all_bonds = []
        
        group_summary = {}
        
        for board_group in self.BOARD_GROUPS:
            url = (
                f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/boardgroups/{board_group}/"
                f"securities.json?iss.dp=comma&iss.meta=off&date={date_str}"
            )
            
            if self.verbose:
                self.logger.debug(f"Запрос облигаций для группы {board_group} на дату {date_str}: {url}")
            
            try:
                time.sleep(self.API_DELAY)
                response = requests.get(url)
                response.raise_for_status()
                json_data = response.json()
                
                if "history" in json_data and "data" in json_data["history"] and json_data["history"]["data"]:
                    columns = json_data["history"]["columns"]
                    
                    # Основная информация
                    secid_idx = columns.index("SECID") if "SECID" in columns else None
                    boardid_idx = columns.index("BOARDID") if "BOARDID" in columns else None
                    shortname_idx = columns.index("SHORTNAME") if "SHORTNAME" in columns else None
                    tradedate_idx = columns.index("TRADEDATE") if "TRADEDATE" in columns else None
                    
                    # Цена и объем
                    close_idx = columns.index("CLOSE") if "CLOSE" in columns else None
                    legalclose_idx = columns.index("LEGALCLOSEPRICE") if "LEGALCLOSEPRICE" in columns else None
                    open_idx = columns.index("OPEN") if "OPEN" in columns else None
                    low_idx = columns.index("LOW") if "LOW" in columns else None
                    high_idx = columns.index("HIGH") if "HIGH" in columns else None
                    waprice_idx = columns.index("WAPRICE") if "WAPRICE" in columns else None
                    volume_idx = columns.index("VOLUME") if "VOLUME" in columns else None
                    value_idx = columns.index("VALUE") if "VALUE" in columns else None
                    
                    # Доходность
                    yield_idx = columns.index("YIELDCLOSE") if "YIELDCLOSE" in columns else None
                    yield_atwap_idx = columns.index("YIELDATWAP") if "YIELDATWAP" in columns else None
                    
                    # Даты и длительность
                    matdate_idx = columns.index("MATDATE") if "MATDATE" in columns else None
                    duration_idx = columns.index("DURATION") if "DURATION" in columns else None
                    offerdate_idx = columns.index("OFFERDATE") if "OFFERDATE" in columns else None
                    
                    # Купоны
                    accint_idx = columns.index("ACCINT") if "ACCINT" in columns else None
                    couponpercent_idx = columns.index("COUPONPERCENT") if "COUPONPERCENT" in columns else None
                    couponvalue_idx = columns.index("COUPONVALUE") if "COUPONVALUE" in columns else None
                    
                    # Номинал
                    facevalue_idx = columns.index("FACEVALUE") if "FACEVALUE" in columns else None
                    faceunit_idx = columns.index("FACEUNIT") if "FACEUNIT" in columns else None
                    
                    group_bonds = []
                    
                    for record in json_data["history"]["data"]:
                        bond_info = {
                            "secid": record[secid_idx] if secid_idx is not None and secid_idx < len(record) else None,
                            "boardid": record[boardid_idx] if boardid_idx is not None and boardid_idx < len(record) else None,
                            "name": record[shortname_idx] if shortname_idx is not None and shortname_idx < len(record) else None,
                            "tradedate": record[tradedate_idx] if tradedate_idx is not None and tradedate_idx < len(record) else date_str,
                            
                            # Цена: сначала CLOSE, если нет, то LEGALCLOSEPRICE
                            "price": record[close_idx] if close_idx is not None and close_idx < len(record) and record[close_idx] is not None 
                                    else record[legalclose_idx] if legalclose_idx is not None and legalclose_idx < len(record) else None,
                            
                            "price_open": record[open_idx] if open_idx is not None and open_idx < len(record) else None,
                            "price_low": record[low_idx] if low_idx is not None and low_idx < len(record) else None,
                            "price_high": record[high_idx] if high_idx is not None and high_idx < len(record) else None,
                            "price_waprice": record[waprice_idx] if waprice_idx is not None and waprice_idx < len(record) else None,
                            
                            "volume": record[volume_idx] if volume_idx is not None and volume_idx < len(record) else None,
                            "value": record[value_idx] if value_idx is not None and value_idx < len(record) else None,
                            
                            "yield": record[yield_idx] if yield_idx is not None and yield_idx < len(record) else None,
                            "yield_atwap": record[yield_atwap_idx] if yield_atwap_idx is not None and yield_atwap_idx < len(record) else None,
                            
                            "maturity_date": record[matdate_idx] if matdate_idx is not None and matdate_idx < len(record) else None,
                            "duration": record[duration_idx] if duration_idx is not None and duration_idx < len(record) else None,
                            "offer_date": record[offerdate_idx] if offerdate_idx is not None and offerdate_idx < len(record) else None,
                            
                            "accint": record[accint_idx] if accint_idx is not None and accint_idx < len(record) else None,
                            "coupon_percent": record[couponpercent_idx] if couponpercent_idx is not None and couponpercent_idx < len(record) else None,
                            "coupon_value": record[couponvalue_idx] if couponvalue_idx is not None and couponvalue_idx < len(record) else None,
                            
                            "face_value": record[facevalue_idx] if facevalue_idx is not None and facevalue_idx < len(record) else None,
                            "face_unit": record[faceunit_idx] if faceunit_idx is not None and faceunit_idx < len(record) else None,
                            
                            "board_group": board_group,
                        }
                        
                        # Преобразуем duration из дней в месяцы, если есть данные
                        if bond_info["duration"] is not None and bond_info["duration"] != 0:
                            bond_info["duration"] = round(bond_info["duration"] / 30 * 100) / 100  # Округляем до 2 знаков
                        
                        if not bond_info["secid"] or not bond_info["name"]:
                            continue
                        
                        group_bonds.append(bond_info)
                    
                    all_bonds.extend(group_bonds)
                    group_summary[board_group] = len(group_bonds)
                    
                else:
                    if self.verbose:
                        self.logger.debug(f"Нет данных для группы {board_group} на дату {date_str}")
                    group_summary[board_group] = 0
                    
            except Exception as e:
                self.logger.error(f"Ошибка при запросе данных для группы {board_group}: {e}")
                group_summary[board_group] = 0
        
        self._save_to_cache(cache_key, all_bonds)
        
        self.logger.info(f"Результаты парсинга облигаций на {date_str}:")
        for group, count in group_summary.items():
            self.logger.info(f"  Группа {group}: {count} облигаций")
        self.logger.info(f"  Всего: {len(all_bonds)} облигаций")
        
        return all_bonds
    
    def _apply_filters(self, bonds, date, yield_range, price_range, duration_range, 
                      volume_threshold, bond_volume_threshold, is_qualified_investors,
                      coupon_type, emitent, issue_year_range, known_coupon_payments):
        """Применить фильтры к списку облигаций"""
        
        if not bonds:
            self.logger.warning("Нет облигаций для применения фильтров")
            return []
            
        filtered_bonds = []
        
        yield_min, yield_max = yield_range
        price_min, price_max = price_range
        duration_min, duration_max = duration_range
        
        if self.verbose:
            self.logger.debug(f"Применяем фильтры к {len(bonds)} облигациям:")
            self.logger.debug(f"  Доходность: от {yield_min}% до {yield_max}%")
            self.logger.debug(f"  Цена: от {price_min}% до {price_max}%")
            self.logger.debug(f"  Дюрация: от {duration_min} до {duration_max} месяцев")
            self.logger.debug(f"  Минимальный объем сделок: {volume_threshold}")
            self.logger.debug(f"  Минимальный совокупный объем: {bond_volume_threshold}")
        
        # Фильтрация
        filtered_count = 0
        for bond in bonds:
            # Пропускаем бонды без базовых параметров
            if bond["yield"] is None or bond["price"] is None or bond["duration"] is None:
                continue
            
            # Применяем базовые фильтры
            if not (yield_min <= bond["yield"] <= yield_max and 
                    price_min <= bond["price"] <= price_max and
                    duration_min <= bond["duration"] <= duration_max):
                continue
                
            filtered_count += 1
            filtered_bonds.append(bond)
        
        self.logger.info(f"Отфильтровано по базовым параметрам: {filtered_count} облигаций")
        
        return filtered_bonds
    
    def _get_last_trading_day(self):
        """Получить дату последнего торгового дня"""
        current_year = 2025
        today = datetime.now()
        
        today = datetime(current_year, today.month, today.day)
        
        if today.weekday() >= 5:  # 5 = суббота, 6 = воскресенье
            days_to_subtract = today.weekday() - 4  # 4 = пятница
            return today - timedelta(days=days_to_subtract)
        
        # Если запрос выполняется рано утром
        elif today.hour < 10:
            if today.weekday() == 0:  # 0 = понедельник
                return today - timedelta(days=3)  # пятница
            return today - timedelta(days=1)
        
        return today
    
    def clear_cache(self, days_old=None):
        """
        Очистить кэш
        
        Args:
            days_old: удалить файлы старше указанного количества дней (None = удалить все)
        """
        if days_old is None:
            shutil.rmtree(self.CACHE_DIR, ignore_errors=True)
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            self.logger.info(f"Кэш полностью очищен")
        else:
            now = datetime.now()
            count = 0
            for filename in os.listdir(self.CACHE_DIR):
                file_path = os.path.join(self.CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (now - file_time).days > days_old:
                        os.remove(file_path)
                        count += 1
            self.logger.info(f"Удалено {count} устаревших файлов кэша")

    
    def filter_existing_data(self, date, **filter_params):
        """
        Фильтровать существующие данные без обращения к API
        
        Args:
            date: дата данных для фильтрации
            **filter_params: параметры фильтрации
            
        Returns:
            Отфильтрованные данные или None, если данные не найдены
        """
        date_str = date.strftime('%Y-%m-%d')
        all_bonds_path = os.path.join(self.BASE_DATA_DIR, date_str, "all.csv")
        
        if not os.path.exists(all_bonds_path):
            self.logger.warning(f"Файл all.csv для даты {date_str} не найден")
            return None
        
        try:
            self.logger.info(f"Чтение данных из {all_bonds_path}")
            all_bonds_df = pd.read_csv(all_bonds_path)
            all_bonds = all_bonds_df.to_dict('records')
            
            filtered_bonds = self._apply_filters(
                all_bonds, 
                date,
                filter_params.get('yield_range', (0, 100)), 
                filter_params.get('price_range', (0, 200)), 
                filter_params.get('duration_range', (0, 360)),
                filter_params.get('volume_threshold', 2000),
                filter_params.get('bond_volume_threshold', 60000),
                filter_params.get('is_qualified_investors', None),
                filter_params.get('coupon_type', None),
                filter_params.get('emitent', None),
                filter_params.get('issue_year_range', None),
                filter_params.get('known_coupon_payments', True)
            )
            
            filter_folder_name = self._create_filter_folder_name(**filter_params)
            filter_dir = os.path.join(self.BASE_DATA_DIR, date_str, filter_folder_name)
            os.makedirs(filter_dir, exist_ok=True)
            
            filtered_path = os.path.join(filter_dir, "filtered.csv")
            if filtered_bonds:
                filtered_df = pd.DataFrame(filtered_bonds)
                filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
                self.logger.info(f"Сохранены отфильтрованные облигации ({len(filtered_bonds)}) в {filtered_path}")
            
            return filtered_bonds
            
        except Exception as e:
            self.logger.error(f"Ошибка при фильтрации существующих данных: {e}")
            return None
        
    def analyze_bonds_continuity(self, start_date, end_date, save_to_csv=True):
        """
        Анализирует непрерывность данных по облигациям в заданном периоде
        
        Args:
            start_date: начало периода
            end_date: конец периода
            save_to_csv: сохранять ли результаты в CSV
            
        Returns:
            Кортеж (полное_пересечение, статистика_по_облигациям)
        """
        self.logger.info(f"Анализ непрерывности данных по облигациям с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                trading_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        self.logger.info(f"Найдено {len(trading_dates)} рабочих дней в указанном периоде")
        
        available_dates = []
        for date_str in trading_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")

            if os.path.exists(all_csv_path):
                available_dates.append(date_str)

        
        self.logger.info(f"Найдено {len(available_dates)} дней с доступными данными")
        
        if not available_dates:
            self.logger.warning("Нет данных для анализа")
            return [], {}
        
        all_bonds = set()
        bond_presence = {}
        
        for date_str in available_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            try:
                df = pd.read_csv(all_csv_path)
                bonds_for_day = set(df['secid'].unique())
                
                all_bonds.update(bonds_for_day)
                
                for bond in bonds_for_day:
                    if bond not in bond_presence:
                        bond_presence[bond] = {date: False for date in available_dates}
                    bond_presence[bond][date_str] = True
                    
            except Exception as e:
                self.logger.error(f"Ошибка при чтении данных за {date_str}: {e}")

        bond_stats = {}
        continuous_bonds = []
        
        for bond in all_bonds:
            days_present = sum(1 for present in bond_presence[bond].values() if present)
            
            presence_percentage = (days_present / len(available_dates)) * 100
            
            is_continuous = all(bond_presence[bond].values())
            
            bond_stats[bond] = {
                'days_present': days_present,
                'total_days': len(available_dates),
                'presence_percentage': presence_percentage,
                'is_continuous': is_continuous
            }

            if is_continuous:
                continuous_bonds.append(bond)
        
        self.logger.info(f"Всего уникальных облигаций: {len(all_bonds)}")
        self.logger.info(f"Облигаций с непрерывными данными: {len(continuous_bonds)} ({(len(continuous_bonds) / len(all_bonds)) * 100:.2f}%)")
        
        if save_to_csv:
            analysis_dir = os.path.join(
                self.BASE_DATA_DIR, 
                f"continuity_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            )
            os.makedirs(analysis_dir, exist_ok=True)
            
            continuous_df = pd.DataFrame({'secid': continuous_bonds})
            continuous_df.to_csv(
                os.path.join(analysis_dir, "continuous_bonds.csv"), 
                index=False
            )
            
            stats_data = []
            for bond, stats in bond_stats.items():
                row = {'secid': bond}
                row.update(stats)
                stats_data.append(row)
            
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values(by=['presence_percentage', 'days_present'], ascending=False)
            stats_df.to_csv(
                os.path.join(analysis_dir, "bond_presence_stats.csv"), 
                index=False
            )
            
            almost_continuous_df = stats_df[stats_df['presence_percentage'] >= 90]
            almost_continuous_df.to_csv(
                os.path.join(analysis_dir, "almost_continuous_bonds.csv"), 
                index=False
            )
            
            summary = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': len(available_dates),
                'total_bonds': len(all_bonds),
                'continuous_bonds': len(continuous_bonds),
                'almost_continuous_bonds': len(almost_continuous_df),
                'continuity_percentage': (len(continuous_bonds) / len(all_bonds)) * 100,
            }
            
            with open(os.path.join(analysis_dir, "summary.json"), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Результаты анализа сохранены в {analysis_dir}")
        
        return continuous_bonds, bond_stats
    
    def analyze_continuity_from_consolidated(dataset_path, save_to_csv=True):
        """
        Analyzes bond continuity from a consolidated dataset file
        
        Args:
            dataset_path: Path to the period_data.csv file
            save_to_csv: Whether to save results to CSV
            
        Returns:
            Tuple of (continuous_bonds, bond_stats)
        """
        print(f"Analyzing continuity from consolidated dataset: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        
        all_dates = sorted(df['date'].unique())
        all_bonds = sorted(df['secid'].unique())
        
        print(f"Found {len(all_dates)} trading days and {len(all_bonds)} unique bonds")
        
        presence_df = pd.pivot_table(
            df, 
            values='price',
            index='date', 
            columns='secid',
            aggfunc='count'
        ).notna()
        
        bond_stats = {}
        continuous_bonds = []
        
        for bond in all_bonds:
            if bond in presence_df.columns:
                days_present = presence_df[bond].sum()
                presence_percentage = (days_present / len(all_dates)) * 100
                is_continuous = (days_present == len(all_dates))
                
                bond_stats[bond] = {
                    'days_present': days_present,
                    'total_days': len(all_dates),
                    'presence_percentage': presence_percentage,
                    'is_continuous': is_continuous
                }
                
                if is_continuous:
                    continuous_bonds.append(bond)
        
        print(f"Found {len(continuous_bonds)} bonds with continuous data ({len(continuous_bonds)/len(all_bonds)*100:.2f}%)")
        
        if save_to_csv:
            output_dir = os.path.dirname(dataset_path)
            
            continuous_df = pd.DataFrame({'secid': continuous_bonds})
            continuous_df.to_csv(os.path.join(output_dir, "continuous_bonds.csv"), index=False)
            
            # Save bond stats
            stats_data = []
            for bond, stats in bond_stats.items():
                row = {'secid': bond}
                row.update(stats)
                stats_data.append(row)
            
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values(by=['presence_percentage', 'days_present'], ascending=False)
            stats_df.to_csv(os.path.join(output_dir, "bond_presence_stats.csv"), index=False)
        
        return continuous_bonds, bond_stats
    
    def find_optimal_threshold_from_consolidated(dataset_path, min_bonds=20, max_threshold=99):
        """
        Find optimal continuity threshold from consolidated dataset
        """
        _, bond_stats = analyze_continuity_from_consolidated(dataset_path, save_to_csv=True)
        
        if not bond_stats:
            print("Could not get continuity statistics")
            return 100
        
        stats_data = []
        for bond, stats in bond_stats.items():
            row = {'secid': bond}
            row.update(stats)
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        
        thresholds = range(max_threshold, 50, -1)
        
        for threshold in thresholds:
            filtered_bonds = stats_df[stats_df['presence_percentage'] >= threshold]
            bond_count = len(filtered_bonds)
            
            print(f"With threshold {threshold}%: {bond_count} bonds available")
            
            if bond_count >= min_bonds:
                print(f"Found optimal threshold: {threshold}%")
                return threshold
        
        print(f"Could not find threshold with {min_bonds} bonds. Maximum available: {len(stats_df[stats_df['presence_percentage'] >= 50])} bonds at 50% threshold")
        return 50

    
    def generate_complete_dataset(self, start_date, end_date, filter_continuous=True):
        """
        Генерирует полный датасет для облигаций за указанный период
        
        Args:
            start_date: начало периода
            end_date: конец периода
            filter_continuous: использовать только облигации с непрерывными данными
            
        Returns:
            DataFrame с полными данными
        """
        self.logger.info(f"Генерация полного датасета с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        continuous_bonds, bond_stats = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=True)
        
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                date_str = current_date.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
                all_csv_path = os.path.join(date_dir, "all.csv")
                if os.path.exists(all_csv_path):
                    trading_dates.append(date_str)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            self.logger.warning("Нет данных для указанного периода")
            return None
        
        all_data = []
        
        for date_str in trading_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            try:
                df = pd.read_csv(all_csv_path)
                
                if filter_continuous:
                    df = df[df['secid'].isin(continuous_bonds)]
                
                df['date'] = date_str
                all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Ошибка при чтении данных за {date_str}: {e}")
        
        if not all_data:
            self.logger.warning("Не удалось собрать данные")
            return None
        
        full_df = pd.concat(all_data, ignore_index=True)
        
        dataset_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"full_dataset_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        os.makedirs(dataset_dir, exist_ok=True)
        
        dataset_type = "continuous" if filter_continuous else "all"
        full_df.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_bonds_dataset.csv"), 
            index=False
        )
        
        pivot_df = full_df[['date', 'secid', 'price', 'yield']].copy()
        
        price_pivot = pivot_df.pivot_table(
            values='price', 
            index='date', 
            columns='secid'
        )
        price_pivot.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_price_pivot.csv")
        )
        
        yield_pivot = pivot_df.pivot_table(
            values='yield', 
            index='date', 
            columns='secid'
        )
        yield_pivot.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_yield_pivot.csv")
        )
        
        self.logger.info(f"Полный датасет создан и сохранен в {dataset_dir}")
        self.logger.info(f"Всего дат: {len(trading_dates)}, облигаций: {len(full_df['secid'].unique())}")
        
        return full_df

    def _get_from_cache(self, key):
        """Получить данные из кэша"""
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.CACHE_DIR, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        file_time = os.path.getmtime(cache_file)
        file_datetime = datetime.fromtimestamp(file_time)
        if (datetime.now() - file_datetime).days > self.cache_ttl_days:
            os.remove(cache_file)
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении из кэша: {e}")
            return None
        

    def _clear_previous_loads(self, start_date, end_date):
        """
        Удаляет предыдущие загрузки для указанного периода
        
        Args:
            start_date: начало периода
            end_date: конец периода
        """
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            
            if os.path.exists(date_dir):
                load_dirs = [d for d in os.listdir(date_dir) 
                            if os.path.isdir(os.path.join(date_dir, d)) and d.startswith('load_')]
                
                for load_dir in load_dirs:
                    load_path = os.path.join(date_dir, load_dir)
                    try:
                        shutil.rmtree(load_path)
                        self.logger.info(f"Удалена директория {load_path}")
                    except Exception as e:
                        self.logger.error(f"Ошибка при удалении директории {load_path}: {e}")
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"Очищены все предыдущие загрузки для периода с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")

    
    def _save_to_cache(self, key, data):
        """Сохранить данные в кэш"""
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.CACHE_DIR, f"{key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=self._json_serial)
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в кэш: {e}")
    
    def _json_serial(self, obj):
        """Сериализатор JSON для нестандартных типов"""
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        raise TypeError(f"Type {type(obj)} not serializable")
    

    def _get_next_load_number(self, date_dir):
        """
        Определяет следующий номер загрузки для указанной даты
        
        Args:
            date_dir: путь к директории даты
            
        Returns:
            Следующий номер загрузки
        """
        if not os.path.exists(date_dir):
            return 1
            
        load_dirs = [d for d in os.listdir(date_dir) 
                    if os.path.isdir(os.path.join(date_dir, d)) and d.startswith('load_')]
        
        load_numbers = []
        for d in load_dirs:
            try:
                num = int(d.split('_')[1])
                load_numbers.append(num)
            except (IndexError, ValueError):
                continue
                
        return max(load_numbers, default=0) + 1

    def parse_interval(self, start_date=None, end_date=None, include_non_trading_days=True, 
                  analyze_continuity=True, load_all_csv=False, clear_previous_loads=False, **filter_params):
        """
        Парсить данные за указанный интервал
        
        Args:
            start_date: начальная дата интервала (по умолчанию 1 января 2024)
            end_date: конечная дата интервала (по умолчанию текущий торговый день)
            include_non_trading_days: включать ли нерабочие дни
            analyze_continuity: выполнять ли анализ непрерывности данных
            load_all_csv: загружать ли все облигации в all.csv (по умолчанию False)
            clear_previous_loads: удалить предыдущие загрузки (по умолчанию False)
            **filter_params: параметры фильтрации
            
        Returns:
            Словарь с датами и списками облигаций
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date is None:
            end_date = self._get_last_trading_day()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.logger.info(f"Запуск парсинга облигаций за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        if clear_previous_loads:
            self._clear_previous_loads(start_date, end_date)
        
        result = self.get_bonds_for_period(
            start_date=start_date,
            end_date=end_date,
            include_non_trading_days=include_non_trading_days,
            load_all_csv=load_all_csv,
            **filter_params
        )
        
        if analyze_continuity and result:
            self.logger.info("Выполняем анализ непрерывности данных...")
            continuous_bonds, bond_stats = self.analyze_bonds_continuity(start_date, end_date)
            
            self.logger.info("Создаем полный датасет...")
            self.generate_complete_dataset(start_date, end_date, filter_continuous=True)
        
        return result

    def find_optimal_threshold(self, start_date, end_date, min_bonds=20, max_threshold=99):
        """
        Находит оптимальный порог непрерывности данных для создания датасета
        
        Args:
            start_date: начало периода
            end_date: конец периода
            min_bonds: минимальное количество облигаций, которое должно быть в датасете
            max_threshold: максимальный порог непрерывности (в процентах)
            
        Returns:
            Оптимальный порог непрерывности (в процентах)
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        self.logger.info(f"Поиск оптимального порога непрерывности для периода с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        _, bond_stats = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=True)
        
        if not bond_stats:
            self.logger.warning("Не удалось получить статистику непрерывности")
            return 100
        
        stats_data = []
        for bond, stats in bond_stats.items():
            row = {'secid': bond}
            row.update(stats)
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        
        thresholds = range(max_threshold, 50, -1)
        
        for threshold in thresholds:
            filtered_bonds = stats_df[stats_df['presence_percentage'] >= threshold]
            bond_count = len(filtered_bonds)
            
            self.logger.info(f"При пороге {threshold}%: доступно {bond_count} облигаций")
            
            if bond_count >= min_bonds:
                self.logger.info(f"Найден оптимальный порог: {threshold}%")
                return threshold
        
        self.logger.warning(f"Не удалось найти порог, при котором доступно {min_bonds} облигаций. Максимум доступно: {len(stats_df[stats_df['presence_percentage'] >= 50])} облигаций при пороге 50%")
        return 50

    def generate_dataset_with_threshold(self, start_date, end_date, continuity_threshold=90):
        """
        Генерирует датасет с облигациями, имеющими указанный порог непрерывности
        
        Args:
            start_date: начало периода
            end_date: конец периода
            continuity_threshold: порог непрерывности (в процентах)
            
        Returns:
            DataFrame с данными или None в случае ошибки
        """
        self.logger.info(f"Генерация датасета с порогом непрерывности {continuity_threshold}% за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        _, bond_stats = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=False)
        
        if not bond_stats:
            self.logger.warning("Не удалось получить статистику непрерывности")
            return None
        
        filtered_bonds = []
        for bond, stats in bond_stats.items():
            if stats['presence_percentage'] >= continuity_threshold:
                filtered_bonds.append(bond)
        
        if not filtered_bonds:
            self.logger.warning(f"Не найдено облигаций с порогом непрерывности {continuity_threshold}%")
            return None
        
        self.logger.info(f"Найдено {len(filtered_bonds)} облигаций с порогом непрерывности не менее {continuity_threshold}%")
        
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                date_str = current_date.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
                all_csv_path = os.path.join(date_dir, "all.csv")
                if os.path.exists(all_csv_path):
                    trading_dates.append(date_str)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            self.logger.warning("Нет данных для указанного периода")
            return None
        
        all_data = []
        
        for date_str in trading_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            try:
                df = pd.read_csv(all_csv_path)
                df = df[df['secid'].isin(filtered_bonds)]
                
                if not df.empty:
                    df['date'] = date_str
                    all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Ошибка при чтении данных за {date_str}: {e}")
        
        if not all_data:
            self.logger.warning("Не удалось собрать данные")
            return None
        
        full_df = pd.concat(all_data, ignore_index=True)
        
        dataset_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"threshold_dataset_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{continuity_threshold}"
        )
        os.makedirs(dataset_dir, exist_ok=True)
        
        full_df.to_csv(
            os.path.join(dataset_dir, "bonds_dataset.csv"), 
            index=False
        )
        
        pivot_df = full_df[['date', 'secid', 'price', 'yield']].copy()
        
        price_pivot = pivot_df.pivot_table(
            values='price', 
            index='date', 
            columns='secid'
        )
        price_pivot.to_csv(
            os.path.join(dataset_dir, "price_pivot.csv")
        )
        
        yield_pivot = pivot_df.pivot_table(
            values='yield', 
            index='date', 
            columns='secid'
        )
        yield_pivot.to_csv(
            os.path.join(dataset_dir, "yield_pivot.csv")
        )
        
        self.logger.info(f"Датасет с порогом непрерывности {continuity_threshold}% создан и сохранен в {dataset_dir}")
        self.logger.info(f"Всего дат: {len(trading_dates)}, облигаций: {len(full_df['secid'].unique())}")
        
        return full_df
    

def run_bond_pipeline(
    base_path,
    start_date,
    end_date=None,
    force_reparse=False,
    min_bonds=20,
    max_threshold=99,
    **filter_params
):
    """
    Comprehensive bond analysis pipeline that's resilient to restarts
    """
    bond_parser = MOEXBondHistoricalParser()
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Running bond analysis for period {start_date_str} to {end_date_str}")
    
    moex_path = os.path.join(base_path, "data/processed_data/BONDS/moex")
    period_path = os.path.join(moex_path, f"period_{start_date_str}_{end_date_str}")
    
    dataset_path = None
    if not force_reparse and os.path.exists(period_path):
        for root, dirs, files in os.walk(period_path):
            if "period_data.csv" in files:
                dataset_path = os.path.join(root, "period_data.csv")
                print(f"Found existing dataset: {dataset_path}")
                break
    
    if dataset_path is None or force_reparse:
        print("Parsing bond data (this may take up to an hour)...")
        result = bond_parser.parse_interval(
            start_date=start_date_str,
            end_date=end_date_str,
            include_non_trading_days=True,
            analyze_continuity=False,
            load_all_csv=True,
            clear_previous_loads=False,
            **filter_params
        )
        
        for root, dirs, files in os.walk(period_path):
            if "period_data.csv" in files:
                dataset_path = os.path.join(root, "period_data.csv")
                print(f"Created new dataset: {dataset_path}")
                break
    
    if dataset_path is None or not os.path.exists(dataset_path):
        print("ERROR: No valid dataset found")
        return {"success": False, "error": "No dataset available"}

    print("Analyzing bond data continuity...")
    
    df = pd.read_csv(dataset_path)
    
    presence_pivot = pd.pivot_table(
        df, values='price', index='date', columns='secid', aggfunc='count'
    ).notna()
    
    all_dates = sorted(df['date'].unique())
    all_bonds = sorted(df['secid'].unique())
    
    bond_stats = {}
    continuous_bonds = []
    
    for bond in all_bonds:
        days_present = presence_pivot[bond].sum() if bond in presence_pivot else 0
        presence_pct = (days_present / len(all_dates)) * 100
        is_continuous = (days_present == len(all_dates))
        
        bond_stats[bond] = {
            'days_present': days_present,
            'total_days': len(all_dates),
            'presence_percentage': presence_pct,
            'is_continuous': is_continuous
        }
        
        if is_continuous:
            continuous_bonds.append(bond)
    
    stats_dir = os.path.dirname(dataset_path)
    stats_df = pd.DataFrame([{'secid': k, **v} for k, v in bond_stats.items()])
    stats_df.to_csv(os.path.join(stats_dir, "bond_presence_stats.csv"), index=False)
    
    thresholds = range(max_threshold, 50, -1)
    optimal_threshold = 50
    
    for threshold in thresholds:
        filtered_bonds = [bond for bond, stats in bond_stats.items() 
                         if stats['presence_percentage'] >= threshold]
        
        print(f"Threshold {threshold}%: {len(filtered_bonds)} bonds available")
        
        if len(filtered_bonds) >= min_bonds:
            optimal_threshold = threshold
            break
    
    print(f"Optimal threshold: {optimal_threshold}%")
    
    filtered_bonds = [bond for bond, stats in bond_stats.items() 
                     if stats['presence_percentage'] >= optimal_threshold]
    
    filtered_df = df[df['secid'].isin(filtered_bonds)]
    
    clean_dir = os.path.join(moex_path, f"bonds_{start_date_str}_{end_date_str}")
    os.makedirs(clean_dir, exist_ok=True)
    
    threshold_dir = os.path.join(clean_dir, f"threshold_{optimal_threshold}")
    os.makedirs(threshold_dir, exist_ok=True)
    
    filtered_path = os.path.join(threshold_dir, "bonds_dataset.csv")
    filtered_df.to_csv(filtered_path, index=False)
    
    price_pivot = filtered_df.pivot_table(values='price', index='date', columns='secid')
    price_pivot.to_csv(os.path.join(threshold_dir, "price_pivot.csv"))
    
    yield_pivot = filtered_df.pivot_table(values='yield', index='date', columns='secid')
    yield_pivot.to_csv(os.path.join(threshold_dir, "yield_pivot.csv"))
    
    print(f"Created filtered dataset with {len(filtered_bonds)} bonds and {len(all_dates)} dates")
    
    return {
        "success": True,
        "dataset_path": filtered_path,
        "optimal_threshold": optimal_threshold,
        "bond_count": len(filtered_bonds),
        "date_count": len(all_dates),
        "period": {
            "start_date": start_date_str,
            "end_date": end_date_str
        }
    }

