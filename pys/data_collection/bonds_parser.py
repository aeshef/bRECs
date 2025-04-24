from datetime import datetime, timedelta
import os
import json
import time
import pandas as pd
import requests
import shutil
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import sys

class Logger:
    def __init__(self, name: str, format: str, store: bool = True):
        self.log = self.__get_logger(name, format)
        self.messages = [] if store else None

    def __get_logger(self, name: str, format: str) -> logging.Logger:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)

        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        return log

    def info(self, message: str):
        if self.messages is not None:
            if message.startswith("\n"):
                self.messages.append("")
            self.messages.append(message)
            if message.endswith("\n"):
                self.messages.append("")

        self.log.info(message)

class MOEXBondHistoricalParser:
    """
    Парсер для получения исторических данных по облигациям Московской биржи
    с возможностью применения различных фильтров.
    """
    
    # Группы торговых площадок для облигаций
    BOARD_GROUPS = [58, 193, 105, 77, 207, 167, 245]
    
    # Задержка между API запросами (секунды)
    API_DELAY = 1.2
    
    # Директории для хранения данных
    CACHE_DIR = "/Users/aeshef/Documents/GitHub/kursach/data/processed_data/BONDS/moex/moex_cache"
    BASE_DATA_DIR = "/Users/aeshef/Documents/GitHub/kursach/data/processed_data/BONDS/moex"
    
    def __init__(self, log=None, use_cache=True, cache_ttl_days=7, verbose=False):
        """
        Инициализация парсера
        
        Args:
            log: объект логгера
            use_cache: использовать ли кэширование запросов
            cache_ttl_days: срок хранения кэша в днях
            verbose: выводить ли подробные логи
        """
        self.log = log
        self.use_cache = use_cache
        self.cache_ttl_days = cache_ttl_days
        self.verbose = verbose
        
        # Создаем директории для хранения данных
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            
        if not os.path.exists(self.BASE_DATA_DIR):
            os.makedirs(self.BASE_DATA_DIR, exist_ok=True)
    
    def get_latest_bonds(self, **filter_params):
        """
        Получить актуальные облигации на последний торговый день
        
        Args:
            **filter_params: параметры фильтрации (см. get_available_bonds)
            
        Returns:
            Список облигаций
        """
        # Используем последний торговый день 2024 года
        last_trading_day = self._get_last_trading_day()
        self._log_info(f"Получение облигаций на последний торговый день: {last_trading_day.strftime('%Y-%m-%d')}")
        
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
                           use_existing_data=True):
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
            
        Returns:
            Список словарей с информацией об облигациях
        """
        if date is None:
            date = self._get_last_trading_day()
        
        date_str = date.strftime('%Y-%m-%d')
        self._log_info(f"Получение списка облигаций на {date_str}")
        
        # Создаем директорию для данной даты
        date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Путь к all.csv для данной даты
        all_bonds_path = os.path.join(date_dir, "all.csv")
        
        # Проверяем, существует ли уже файл all.csv и можно ли его использовать
        all_bonds = None
        if use_existing_data and os.path.exists(all_bonds_path):
            try:
                self._log_info(f"Используем существующий файл данных: {all_bonds_path}")
                all_bonds_df = pd.read_csv(all_bonds_path)
                all_bonds = all_bonds_df.to_dict('records')
            except Exception as e:
                self._log_info(f"Ошибка при чтении существующего файла: {e}")
                all_bonds = None
        
        # Если нет существующих данных или их нельзя использовать, запрашиваем с API
        if all_bonds is None:
            all_bonds = self._get_all_bonds_for_date(date)
            # Сохраняем полные данные в all.csv
            if all_bonds:
                all_bonds_df = pd.DataFrame(all_bonds)
                all_bonds_df.to_csv(all_bonds_path, index=False, encoding='utf-8-sig')
                self._log_info(f"Сохранены все облигации ({len(all_bonds)}) в {all_bonds_path}")
        
        # Формируем имя папки для фильтрованных данных
        filter_folder_name = self._create_filter_folder_name(
            yield_range, price_range, duration_range, volume_threshold, 
            bond_volume_threshold, is_qualified_investors, coupon_type, 
            emitent, issue_year_range, known_coupon_payments
        )
        
        filter_dir = os.path.join(date_dir, filter_folder_name)
        os.makedirs(filter_dir, exist_ok=True)
        
        # Применяем фильтры
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
        
        # Сохраняем отфильтрованные данные
        filtered_path = os.path.join(filter_dir, "filtered.csv")
        if filtered_bonds:
            filtered_df = pd.DataFrame(filtered_bonds)
            filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
            self._log_info(f"Сохранены отфильтрованные облигации ({len(filtered_bonds)}) в {filtered_path}")
        
        # Также сохраняем параметры фильтрации в JSON
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
            "filtered_bonds": len(filtered_bonds)
        }
        
        with open(os.path.join(filter_dir, "params.json"), 'w', encoding='utf-8') as f:
            json.dump(filter_params, f, ensure_ascii=False, indent=2)
        
        self._log_info(f"Найдено {len(filtered_bonds)} облигаций после фильтрации")
        
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
        
        # Проверяем существование исходной директории
        if not os.path.exists(source_dir):
            self._log_info(f"Исходная директория {source_dir} не существует")
            return
        
        # Создаем целевую директорию
        os.makedirs(target_dir, exist_ok=True)
        
        # Копируем all.csv
        source_all_csv = os.path.join(source_dir, "all.csv")
        target_all_csv = os.path.join(target_dir, "all.csv")
        
        if os.path.exists(source_all_csv):
            try:
                # Читаем исходный CSV
                df = pd.read_csv(source_all_csv)
                
                # Сохраняем в целевую директорию
                df.to_csv(target_all_csv, index=False, encoding='utf-8-sig')
                self._log_info(f"Скопированы данные из {source_all_csv} в {target_all_csv}")
            except Exception as e:
                self._log_info(f"Ошибка при копировании данных: {e}")
        
    def get_bonds_for_period(self, start_date, end_date, include_non_trading_days=True, **filter_params):
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
        # Проверка корректности дат
        last_trading_day = self._get_last_trading_day()
        
        # Логируем входные параметры
        self._log_info(f"Запрошен период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Корректируем даты, если они в будущем
        if start_date > last_trading_day:
            start_date = last_trading_day
            self._log_info(f"Скорректирована начальная дата на {start_date.strftime('%Y-%m-%d')}")
            
        if end_date > last_trading_day:
            end_date = last_trading_day
            self._log_info(f"Скорректирована конечная дата на {end_date.strftime('%Y-%m-%d')}")
        
        # Если конечная дата раньше начальной, меняем их местами
        if end_date < start_date:
            start_date, end_date = end_date, start_date
            self._log_info("Даты начала и конца периода были поменяны местами")
        
        self._log_info(f"Получение облигаций за период с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Формируем имя папки для периода
        period_dir_name = f"period_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Создаем директорию для периода
        period_dir = os.path.join(self.BASE_DATA_DIR, period_dir_name)
        os.makedirs(period_dir, exist_ok=True)
        
        # Формируем имя папки для фильтрованных данных
        filter_folder_name = self._create_filter_folder_name(**filter_params)
        filter_dir = os.path.join(period_dir, filter_folder_name)
        os.makedirs(filter_dir, exist_ok=True)
        
        result = {}
        current_date = start_date
        
        # Для отслеживания последнего торгового дня
        last_trading_data = None
        last_trading_date = None
        
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            is_trading_day = current_date.weekday() < 5  # 0-4: понедельник-пятница
            
            self._log_info(f"Обработка даты: {date_key} ({'рабочий день' if is_trading_day else 'выходной'})")
            
            if is_trading_day:
                # Рабочий день - делаем запрос к API
                bonds = self.get_available_bonds(date=current_date, **filter_params)
                
                if bonds:
                    result[date_key] = bonds
                    self._log_info(f"На {date_key} найдено {len(bonds)} облигаций")
                    
                    # Обновляем последний торговый день
                    last_trading_data = bonds
                    last_trading_date = current_date
                else:
                    self._log_info(f"На {date_key} облигаций не найдено, возможно праздник или выходной")
                    
                    # Проверяем наличие данных за последний торговый день
                    if include_non_trading_days and last_trading_data:
                        result[date_key] = last_trading_data
                        self._log_info(f"Использованы данные последнего торгового дня ({last_trading_date.strftime('%Y-%m-%d')}) для {date_key}")
                        
                        # Создаем директорию для данной даты и копируем все файлы с последнего торгового дня
                        self._duplicate_trading_day_data(last_trading_date, current_date)
                        
            elif include_non_trading_days and last_trading_data:
                # Выходной день - используем данные последнего торгового дня
                result[date_key] = last_trading_data
                self._log_info(f"Использованы данные последнего торгового дня ({last_trading_date.strftime('%Y-%m-%d')}) для {date_key}")
                
                # Создаем директорию для данной даты и копируем все файлы с последнего торгового дня
                self._duplicate_trading_day_data(last_trading_date, current_date)
                
            current_date += timedelta(days=1)
        
        # Сохраняем агрегированные результаты
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
            self._log_info(f"Сохранены результаты за период в {period_file}")
            
            # Создаем сводку по дням
            summary = {}
            for date, bonds in result.items():
                summary[date] = len(bonds)
                
            summary_df = pd.DataFrame(list(summary.items()), columns=['date', 'bond_count'])
            summary_df.to_csv(os.path.join(filter_dir, "period_summary.csv"), index=False)
        
        # Также сохраняем параметры фильтрации в JSON
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
        
        # Добавляем базовые параметры
        parts.append(f"yieldrange_{yield_range[0]}_{yield_range[1]}")
        parts.append(f"pricerange_{price_range[0]}_{price_range[1]}")
        parts.append(f"durationrange_{duration_range[0]}_{duration_range[1]}")
        
        # Добавляем пороги объема
        parts.append(f"volthresh_{volume_threshold}")
        parts.append(f"bondvolthresh_{bond_volume_threshold}")
        
        # Добавляем опциональные фильтры
        if is_qualified_investors is not None:
            parts.append(f"qualified_{str(is_qualified_investors).lower()}")
            
        if coupon_type is not None:
            parts.append(f"coupontype_{coupon_type}")
            
        if emitent is not None:
            # Очищаем имя эмитента для безопасного использования в имени папки
            safe_emitent = "".join(c if c.isalnum() else "_" for c in emitent.lower())
            parts.append(f"emitent_{safe_emitent}")
            
        if issue_year_range is not None:
            parts.append(f"issueyear_{issue_year_range[0]}_{issue_year_range[1]}")
            
        if known_coupon_payments:
            parts.append("known_coupons")
        
        # Создаем имя папки, используя параметры фильтрации
        folder_name = "__".join(parts)
        
        # Ограничиваем длину имени папки
        if len(folder_name) > 200:
            folder_name = folder_name[:200]
            
        return folder_name
    
    def _get_all_bonds_for_date(self, date):
        """
        Получить все доступные облигации на указанную дату
        """
        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"all_bonds_{date_str}"
        
        # Проверяем кэш
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            self._log_info(f"Используем кэшированные данные для {date_str}")
            return cached_data
        
        all_bonds = []
        
        # Записываем информацию о группах и количестве облигаций
        group_summary = {}
        
        for board_group in self.BOARD_GROUPS:
            # Для исторических данных используем правильный эндпоинт
            url = (
                f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/boardgroups/{board_group}/"
                f"securities.json?iss.dp=comma&iss.meta=off&date={date_str}"
            )
            
            if self.verbose:
                self._log_info(f"Запрос облигаций для группы {board_group} на дату {date_str}: {url}")
            
            try:
                time.sleep(self.API_DELAY)
                response = requests.get(url)
                response.raise_for_status()
                json_data = response.json()
                
                # Проверяем формат данных в ответе
                if "history" in json_data and "data" in json_data["history"] and json_data["history"]["data"]:
                    # Получаем колонки для правильного индексирования данных
                    columns = json_data["history"]["columns"]
                    
                    # Находим индексы всех возможных колонок
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
                    
                    # Добавляем облигации из этой группы
                    group_bonds = []
                    
                    for record in json_data["history"]["data"]:
                        # Получаем все доступные значения из записи
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
                        
                        # Пропускаем записи без ключевых данных
                        if not bond_info["secid"] or not bond_info["name"]:
                            continue
                        
                        group_bonds.append(bond_info)
                    
                    # Добавляем в общий список
                    all_bonds.extend(group_bonds)
                    group_summary[board_group] = len(group_bonds)
                    
                else:
                    if self.verbose:
                        self._log_info(f"Нет данных для группы {board_group} на дату {date_str}")
                    group_summary[board_group] = 0
                    
            except Exception as e:
                self._log_info(f"Ошибка при запросе данных для группы {board_group}: {e}")
                group_summary[board_group] = 0
        
        # Сохраняем в кэш
        self._save_to_cache(cache_key, all_bonds)
        
        # Логируем результаты по группам
        self._log_info(f"Результаты парсинга облигаций на {date_str}:")
        for group, count in group_summary.items():
            self._log_info(f"  Группа {group}: {count} облигаций")
        self._log_info(f"  Всего: {len(all_bonds)} облигаций")
        
        return all_bonds
    
    def _apply_filters(self, bonds, date, yield_range, price_range, duration_range, 
                      volume_threshold, bond_volume_threshold, is_qualified_investors,
                      coupon_type, emitent, issue_year_range, known_coupon_payments):
        """Применить фильтры к списку облигаций"""
        
        if not bonds:
            self._log_info("Нет облигаций для применения фильтров")
            return []
            
        filtered_bonds = []
        
        yield_min, yield_max = yield_range
        price_min, price_max = price_range
        duration_min, duration_max = duration_range
        
        # Логируем параметры фильтрации
        if self.verbose:
            self._log_info(f"Применяем фильтры к {len(bonds)} облигациям:")
            self._log_info(f"  Доходность: от {yield_min}% до {yield_max}%")
            self._log_info(f"  Цена: от {price_min}% до {price_max}%")
            self._log_info(f"  Дюрация: от {duration_min} до {duration_max} месяцев")
            self._log_info(f"  Минимальный объем сделок: {volume_threshold}")
            self._log_info(f"  Минимальный совокупный объем: {bond_volume_threshold}")
        
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
        
        self._log_info(f"Отфильтровано по базовым параметрам: {filtered_count} облигаций")
        
        return filtered_bonds
    
    def _get_last_trading_day(self):
        """Получить дату последнего торгового дня"""
        # Явно используем 2024 год для исторических данных
        current_year = 2025
        today = datetime.now()
        
        today = datetime(current_year, today.month, today.day)
        
        # Если сегодня выходной
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
            # Удаляем всю директорию и создаем заново
            shutil.rmtree(self.CACHE_DIR, ignore_errors=True)
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            self._log_info(f"Кэш полностью очищен")
        else:
            # Удаляем только старые файлы
            now = datetime.now()
            count = 0
            for filename in os.listdir(self.CACHE_DIR):
                file_path = os.path.join(self.CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (now - file_time).days > days_old:
                        os.remove(file_path)
                        count += 1
            self._log_info(f"Удалено {count} устаревших файлов кэша")

    
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
            self._log_info(f"Файл all.csv для даты {date_str} не найден")
            return None
        
        try:
            self._log_info(f"Чтение данных из {all_bonds_path}")
            all_bonds_df = pd.read_csv(all_bonds_path)
            all_bonds = all_bonds_df.to_dict('records')
            
            # Применяем фильтры к загруженным данным
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
            
            # Формируем имя папки для фильтрованных данных
            filter_folder_name = self._create_filter_folder_name(**filter_params)
            filter_dir = os.path.join(self.BASE_DATA_DIR, date_str, filter_folder_name)
            os.makedirs(filter_dir, exist_ok=True)
            
            # Сохраняем отфильтрованные данные
            filtered_path = os.path.join(filter_dir, "filtered.csv")
            if filtered_bonds:
                filtered_df = pd.DataFrame(filtered_bonds)
                filtered_df.to_csv(filtered_path, index=False, encoding='utf-8-sig')
                self._log_info(f"Сохранены отфильтрованные облигации ({len(filtered_bonds)}) в {filtered_path}")
            
            return filtered_bonds
            
        except Exception as e:
            self._log_info(f"Ошибка при фильтрации существующих данных: {e}")
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
        self._log_info(f"Анализ непрерывности данных по облигациям с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Получаем список всех дат в указанном периоде (только рабочие дни)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Только рабочие дни (0-4: понедельник-пятница)
                trading_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        self._log_info(f"Найдено {len(trading_dates)} рабочих дней в указанном периоде")
        
        # Проверяем наличие данных за каждый день
        available_dates = []
        for date_str in trading_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            if os.path.exists(all_csv_path):
                available_dates.append(date_str)
        
        self._log_info(f"Найдено {len(available_dates)} дней с доступными данными")
        
        # Если нет данных, возвращаем пустые результаты
        if not available_dates:
            self._log_info("Нет данных для анализа")
            return [], {}
        
        # Собираем информацию о всех облигациях
        all_bonds = set()
        bond_presence = {}  # {secid: {date1: True, date2: True, ...}}
        
        for date_str in available_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            try:
                df = pd.read_csv(all_csv_path)
                bonds_for_day = set(df['secid'].unique())
                
                # Добавляем новые облигации в общий список
                all_bonds.update(bonds_for_day)
                
                # Обновляем статистику присутствия
                for bond in bonds_for_day:
                    if bond not in bond_presence:
                        bond_presence[bond] = {date: False for date in available_dates}
                    bond_presence[bond][date_str] = True
                    
            except Exception as e:
                self._log_info(f"Ошибка при чтении данных за {date_str}: {e}")
        
        # Подсчитываем статистику для каждой облигации
        bond_stats = {}
        continuous_bonds = []
        
        for bond in all_bonds:
            # Подсчитываем количество дней, когда облигация присутствовала
            days_present = sum(1 for present in bond_presence[bond].values() if present)
            
            # Вычисляем процент присутствия
            presence_percentage = (days_present / len(available_dates)) * 100
            
            # Проверяем, есть ли облигация во всех датасетах
            is_continuous = all(bond_presence[bond].values())
            
            # Добавляем статистику
            bond_stats[bond] = {
                'days_present': days_present,
                'total_days': len(available_dates),
                'presence_percentage': presence_percentage,
                'is_continuous': is_continuous
            }
            
            # Если облигация присутствует во всех датасетах, добавляем в список непрерывных
            if is_continuous:
                continuous_bonds.append(bond)
        
        self._log_info(f"Всего уникальных облигаций: {len(all_bonds)}")
        self._log_info(f"Облигаций с непрерывными данными: {len(continuous_bonds)} ({(len(continuous_bonds) / len(all_bonds)) * 100:.2f}%)")
        
        # Создаем директорию для сохранения результатов анализа
        if save_to_csv:
            analysis_dir = os.path.join(
                self.BASE_DATA_DIR, 
                f"continuity_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            )
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Сохраняем список облигаций с непрерывными данными
            continuous_df = pd.DataFrame({'secid': continuous_bonds})
            continuous_df.to_csv(
                os.path.join(analysis_dir, "continuous_bonds.csv"), 
                index=False
            )
            
            # Сохраняем детальную статистику по всем облигациям
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
            
            # Создаем отдельный файл для почти непрерывных облигаций (>90%)
            almost_continuous_df = stats_df[stats_df['presence_percentage'] >= 90]
            almost_continuous_df.to_csv(
                os.path.join(analysis_dir, "almost_continuous_bonds.csv"), 
                index=False
            )
            
            # Сохраняем сводку
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
            
            self._log_info(f"Результаты анализа сохранены в {analysis_dir}")
        
        return continuous_bonds, bond_stats

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
        self._log_info(f"Генерация полного датасета с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Сначала анализируем непрерывность данных
        continuous_bonds, bond_stats = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=True)
        
        # Получаем список дат в периоде (только рабочие дни)
        trading_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Только рабочие дни
                date_str = current_date.strftime('%Y-%m-%d')
                # Проверяем наличие данных
                date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
                all_csv_path = os.path.join(date_dir, "all.csv")
                if os.path.exists(all_csv_path):
                    trading_dates.append(date_str)
            current_date += timedelta(days=1)
        
        if not trading_dates:
            self._log_info("Нет данных для указанного периода")
            return None
        
        # Собираем датасет
        all_data = []
        
        for date_str in trading_dates:
            date_dir = os.path.join(self.BASE_DATA_DIR, date_str)
            all_csv_path = os.path.join(date_dir, "all.csv")
            
            try:
                df = pd.read_csv(all_csv_path)
                
                # Фильтруем только непрерывные облигации, если требуется
                if filter_continuous:
                    df = df[df['secid'].isin(continuous_bonds)]
                
                # Добавляем дату
                df['date'] = date_str
                
                all_data.append(df)
                
            except Exception as e:
                self._log_info(f"Ошибка при чтении данных за {date_str}: {e}")
        
        if not all_data:
            self._log_info("Не удалось собрать данные")
            return None
        
        # Объединяем все данные
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Создаем директорию для полного датасета
        dataset_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"full_dataset_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Сохраняем полный датасет
        dataset_type = "continuous" if filter_continuous else "all"
        full_df.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_bonds_dataset.csv"), 
            index=False
        )
        
        # Создаем сводную таблицу по датам и облигациям (pivot)
        # Берем только нужные колонки для pivot table
        pivot_df = full_df[['date', 'secid', 'price', 'yield']].copy()
        
        # Создаем сводную таблицу по ценам
        price_pivot = pivot_df.pivot_table(
            values='price', 
            index='date', 
            columns='secid'
        )
        price_pivot.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_price_pivot.csv")
        )
        
        # Создаем сводную таблицу по доходностям
        yield_pivot = pivot_df.pivot_table(
            values='yield', 
            index='date', 
            columns='secid'
        )
        yield_pivot.to_csv(
            os.path.join(dataset_dir, f"{dataset_type}_yield_pivot.csv")
        )
        
        self._log_info(f"Полный датасет создан и сохранен в {dataset_dir}")
        self._log_info(f"Всего дат: {len(trading_dates)}, облигаций: {len(full_df['secid'].unique())}")
        
        return full_df
    

    def generate_dataset_with_threshold(self, start_date, end_date, continuity_threshold=95, recalculate_analysis=False):
        """
        Генерирует датасет для облигаций, которые торговались в указанном проценте дней
        
        Args:
            start_date: начало периода
            end_date: конец периода
            continuity_threshold: порог непрерывности в процентах (по умолчанию 95%)
            recalculate_analysis: пересчитывать ли анализ непрерывности
            
        Returns:
            DataFrame с полными данными для облигаций, превышающих порог непрерывности
        """
        self._log_info(f"Генерация датасета с порогом непрерывности {continuity_threshold}% с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        # Директория с результатами анализа
        analysis_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"continuity_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        
        # Директория для результатов
        dataset_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"threshold_dataset_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{continuity_threshold}"
        )
        os.makedirs(dataset_dir, exist_ok=True)
        
        bond_stats_file = os.path.join(analysis_dir, "bond_presence_stats.csv")
        
        # Проверяем, существуют ли уже результаты анализа
        if not os.path.exists(bond_stats_file) or recalculate_analysis:
            self._log_info("Анализ непрерывности не найден или требуется пересчет, выполняем анализ...")
            _, _ = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=True)
        
        # Загружаем статистику по облигациям
        if not os.path.exists(bond_stats_file):
            self._log_info(f"Файл статистики {bond_stats_file} не найден")
            return None
        
        # Читаем статистику
        bond_stats_df = pd.read_csv(bond_stats_file)
        
        # Фильтруем облигации по порогу непрерывности
        threshold_bonds = bond_stats_df[bond_stats_df['presence_percentage'] >= continuity_threshold]['secid'].tolist()
        
        if not threshold_bonds:
            self._log_info(f"Не найдено облигаций с непрерывностью >= {continuity_threshold}%")
            return None
        
        self._log_info(f"Найдено {len(threshold_bonds)} облигаций с непрерывностью >= {continuity_threshold}%")
        
        # Получаем список доступных дат в формате строк YYYY-MM-DD
        available_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Только рабочие дни
                date_str = current_date.strftime('%Y-%m-%d')
                # Проверяем наличие данных
                all_csv_path = os.path.join(self.BASE_DATA_DIR, date_str, "all.csv")
                if os.path.exists(all_csv_path):
                    available_dates.append(date_str)
            current_date += timedelta(days=1)
        
        # Сохраняем список облигаций, превышающих порог
        threshold_df = pd.DataFrame({'secid': threshold_bonds})
        threshold_df.to_csv(os.path.join(dataset_dir, f"bonds_above_{continuity_threshold}pct.csv"), index=False)
        
        # Создаем пустые DataFrame для загрузки данных
        price_data = pd.DataFrame(index=available_dates, columns=threshold_bonds)
        yield_data = pd.DataFrame(index=available_dates, columns=threshold_bonds)
        
        # Загружаем данные по каждой дате без загрузки всего датасета
        for date_str in available_dates:
            try:
                # Загружаем только нужные данные (без полной загрузки всех данных)
                all_csv_path = os.path.join(self.BASE_DATA_DIR, date_str, "all.csv")
                # Используем оптимизированную загрузку с выбором только нужных колонок
                day_data = pd.read_csv(all_csv_path, usecols=['secid', 'price', 'yield'])
                
                # Фильтруем только нужные облигации
                day_data = day_data[day_data['secid'].isin(threshold_bonds)]
                
                # Заполняем данные в pivoted DataFrame
                for _, row in day_data.iterrows():
                    secid = row['secid']
                    if secid in threshold_bonds:
                        if 'price' in row and not pd.isna(row['price']):
                            price_data.loc[date_str, secid] = row['price']
                        if 'yield' in row and not pd.isna(row['yield']):
                            yield_data.loc[date_str, secid] = row['yield']
                
            except Exception as e:
                self._log_info(f"Ошибка при загрузке данных за {date_str}: {e}")
        
        # Заполняем пропуски методом forward fill (используя последнее известное значение)
        price_data_filled = price_data.ffill()
        yield_data_filled = yield_data.ffill()
        
        # Сохраняем pivot таблицы
        price_data.to_csv(os.path.join(dataset_dir, f"price_pivot_raw.csv"))
        yield_data.to_csv(os.path.join(dataset_dir, f"yield_pivot_raw.csv"))
        
        # Сохраняем заполненные pivot таблицы
        price_data_filled.to_csv(os.path.join(dataset_dir, f"price_pivot_filled.csv"))
        yield_data_filled.to_csv(os.path.join(dataset_dir, f"yield_pivot_filled.csv"))
        
        # Преобразуем pivot tables обратно в длинный формат для создания полного датасета
        full_data = []
        
        for date_str in available_dates:
            for secid in threshold_bonds:
                price = price_data_filled.loc[date_str, secid]
                yield_val = yield_data_filled.loc[date_str, secid]
                
                if not pd.isna(price) or not pd.isna(yield_val):
                    full_data.append({
                        'date': date_str,
                        'secid': secid,
                        'price': price,
                        'yield': yield_val
                    })
        
        # Создаем полный датасет
        full_df = pd.DataFrame(full_data)
        full_df.to_csv(os.path.join(dataset_dir, f"full_dataset_{continuity_threshold}pct.csv"), index=False)
        
        # Создаем сводку
        summary = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'continuity_threshold': continuity_threshold,
            'total_days': len(available_dates),
            'total_bonds_analyzed': len(bond_stats_df),
            'bonds_above_threshold': len(threshold_bonds),
            'threshold_percentage': (len(threshold_bonds) / len(bond_stats_df)) * 100 if len(bond_stats_df) > 0 else 0,
        }
        
        with open(os.path.join(dataset_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self._log_info(f"Датасет с порогом непрерывности {continuity_threshold}% создан и сохранен в {dataset_dir}")
        self._log_info(f"Всего дат: {len(available_dates)}, облигаций: {len(threshold_bonds)}")
        
        return full_df

    def find_optimal_threshold(self, start_date, end_date, min_bonds=10, max_threshold=99):
        """
        Найти оптимальный порог непрерывности, чтобы получить нужное количество облигаций
        
        Args:
            start_date: начало периода
            end_date: конец периода
            min_bonds: минимальное желаемое количество облигаций
            max_threshold: максимальный допустимый порог в процентах
            
        Returns:
            Оптимальный порог непрерывности в процентах
        """
        self._log_info(f"Поиск оптимального порога непрерывности для получения минимум {min_bonds} облигаций")
        
        # Сначала проверяем, есть ли результаты анализа непрерывности
        analysis_dir = os.path.join(
            self.BASE_DATA_DIR, 
            f"continuity_analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        bond_stats_file = os.path.join(analysis_dir, "bond_presence_stats.csv")
        
        if not os.path.exists(bond_stats_file):
            self._log_info(f"Файл статистики не найден, выполняем анализ непрерывности...")
            _, _ = self.analyze_bonds_continuity(start_date, end_date, save_to_csv=True)
        
        # Если всё еще нет файла, значит что-то пошло не так
        if not os.path.exists(bond_stats_file):
            self._log_info(f"Не удалось создать файл статистики")
            return None
        
        # Загружаем статистику
        bond_stats_df = pd.read_csv(bond_stats_file)
        
        # Сортируем по проценту непрерывности от высокого к низкому
        bond_stats_df = bond_stats_df.sort_values(by='presence_percentage', ascending=False)
        
        # Ищем оптимальный порог
        for threshold in range(max_threshold, 0, -1):
            bonds_count = len(bond_stats_df[bond_stats_df['presence_percentage'] >= threshold])
            self._log_info(f"Порог {threshold}%: {bonds_count} облигаций")
            
            if bonds_count >= min_bonds:
                self._log_info(f"Найден оптимальный порог: {threshold}% с {bonds_count} облигациями")
                return threshold
        
        # Если нет такого порога, возвращаем минимальный
        self._log_info(f"Не найден порог с {min_bonds} облигациями, возвращаем минимальный")
        min_threshold = 1
        bonds_count = len(bond_stats_df[bond_stats_df['presence_percentage'] >= min_threshold])
        self._log_info(f"Порог {min_threshold}%: {bonds_count} облигаций")
        return min_threshold


    
    def _get_from_cache(self, key):
        """Получить данные из кэша"""
        if not self.use_cache:
            return None
            
        cache_file = os.path.join(self.CACHE_DIR, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
            
        # Проверяем TTL кэша
        file_time = os.path.getmtime(cache_file)
        file_datetime = datetime.fromtimestamp(file_time)
        if (datetime.now() - file_datetime).days > self.cache_ttl_days:
            # Если кэш устарел, удаляем и возвращаем None
            os.remove(cache_file)
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, key, data):
        """Сохранить данные в кэш"""
        if not self.use_cache:
            return
            
        cache_file = os.path.join(self.CACHE_DIR, f"{key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=self._json_serial)
        except Exception as e:
            self._log_info(f"Ошибка при сохранении в кэш: {e}")
    
    def _json_serial(self, obj):
        """Сериализатор JSON для нестандартных типов"""
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _log_info(self, message):
        """Логирование сообщения"""
        if self.log:
            self.log.info(message)
        else:
            print(message)

    def parse_from_2024_to_now(self, include_non_trading_days=True, analyze_continuity=True, **filter_params):
        """
        Парсить данные с начала 2024 года по текущий день
        
        Args:
            include_non_trading_days: включать ли нерабочие дни
            analyze_continuity: выполнять ли анализ непрерывности данных
            **filter_params: параметры фильтрации
            
        Returns:
            Словарь с датами и списками облигаций
        """
        start_date = datetime(2024, 1, 1)
        end_date = self._get_last_trading_day()
        
        self._log_info(f"Запуск массового парсинга данных с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        
        result = self.get_bonds_for_period(
            start_date=start_date,
            end_date=end_date,
            include_non_trading_days=include_non_trading_days,
            **filter_params
        )
        
        # Если требуется, анализируем непрерывность данных и создаем полный датасет
        if analyze_continuity and result:
            self._log_info("Выполняем анализ непрерывности данных...")
            continuous_bonds, _ = self.analyze_bonds_continuity(start_date, end_date)
            
            self._log_info("Создаем полный датасет...")
            self.generate_complete_dataset(start_date, end_date, filter_continuous=True)
        
        return result

