import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import traceback

from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH
from pys.data_collection.bonds_processor import run_pipeline_bonds_processor
from pys.data_collection.kbd_analyzer import KBDAnalyzer
from pys.data_collection.kbd import KBDDownloader  # Исправлен импорт

def run_bond_selection_with_kbd(
    base_path=BASE_PATH,
    dataset_path=None,
    n_bonds=5,
    min_bonds=3,  # Минимальное желаемое количество облигаций
    weighting_strategy=None,  # None для автоматического определения на основе КБД
    portfolio_stability=0.7,
    use_kbd_recommendations=True,
    override_params=None,
    start_date=None,
    end_date=None,
    update_kbd_data=True,
    strategy_profile='balanced',  # 'conservative', 'balanced', 'aggressive'
    kbd_yield_adjustment=-2.0,    # Корректировка рекомендованной доходности
    kbd_duration_flexibility=1.5,  # Множитель для расширения диапазона дюрации
    max_adjustment_iterations=3,   # Максимальное число итераций адаптивной настройки
    excluded_issuers=None,         # Исключаемые эмитенты
    output_format='all',           # 'all', 'portfolio', 'metrics', 'viz'
    kbd_data=None,                  # Предварительно загруженные данные КБД
    output_dirs=None
):
    """
    Полный пайплайн для выбора облигаций с учетом кривой бескупонной доходности.

    Args:
        base_path (str): базовый путь проекта.
        dataset_path (str): путь к датасету облигаций.
        n_bonds (int): целевое количество облигаций в портфеле.
        min_bonds (int): минимальное количество облигаций для диверсификации.
        weighting_strategy (str, optional): стратегия взвешивания (None для автовыбора на основе КБД).
        portfolio_stability (float): стабильность портфеля (0.0-1.0).
        use_kbd_recommendations (bool): использовать ли рекомендации на основе КБД.
        override_params (dict): словарь для переопределения параметров.
        start_date (str or datetime, optional): начальная дата для КБД (если None, используется год назад).
        end_date (str or datetime, optional): конечная дата для КБД (если None, используется текущая дата).
        update_kbd_data (bool, optional): обновлять ли данные КБД.
        strategy_profile (str): профиль стратегии ('conservative', 'balanced', 'aggressive').
        kbd_yield_adjustment (float): корректировка минимальной доходности (± процентные пункты).
        kbd_duration_flexibility (float): множитель для расширения диапазона дюрации.
        max_adjustment_iterations (int): максимальное число итераций адаптивной настройки.
        excluded_issuers (list, optional): список исключаемых эмитентов.
        output_format (str): формат выходных данных.
        kbd_data (DataFrame, optional): предварительно загруженные данные КБД.

    Returns:
        dict: Результаты работы пайплайна.
    """

    logger = BaseLogger('BondsKBDPipeline').logger
    logger.info("Запуск пайплайна выбора облигаций с учетом КБД")
    
    # Создаем идентификатор запуска для сохранения файлов
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Установка базовой директории для KBD
    kbd_dir = f"{base_path}/data/processed_data/BONDS/kbd"
    
    data_dir = output_dirs.get('data', os.path.join(kbd_dir, 'data'))
    results_dir = output_dirs.get('results', os.path.join(kbd_dir, 'results'))
    viz_dir = output_dirs.get('viz', os.path.join(kbd_dir, 'viz'))
    analysis_dir = output_dirs.get('analysis', os.path.join(kbd_dir, 'analysis'))
    portfolios_dir = output_dirs.get('portfolios', os.path.join(kbd_dir, 'portfolios'))
    reports_dir = output_dirs.get('reports', os.path.join(kbd_dir, 'reports'))
    
    # Создаем все нужные директории
    for dir_path in [kbd_dir, data_dir, results_dir, viz_dir, analysis_dir, portfolios_dir, reports_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Установка дат для КБД, если не указаны
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    if output_dirs is None:
        output_dirs = {}    
    
    # Применяем настройки для выбранного профиля стратегии
    strategy_settings = _get_strategy_profile_settings(strategy_profile)
    
    # Шаг 1: Загрузка данных КБД
    kbd_results = {}
    filter_params = {}
    recommended_weighting = None
    
    if use_kbd_recommendations:
        try:
            # Проверяем, предоставлены ли данные КБД извне
            if kbd_data is not None and not kbd_data.empty:
                logger.info(f"Используем предоставленные данные КБД: {len(kbd_data)} записей")
            else:
                # Загружаем данные КБД если они не предоставлены
                from pys.data_collection.kbd import KBDDownloader
                
                downloader = KBDDownloader(output_dir=kbd_dir)
                
                if update_kbd_data:
                    # Загружаем актуальные данные с MOEX API
                    logger.info(f"Загружаем данные КБД с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
                    kbd_data = downloader.get_kbd(start_date, end_date)
                    
                    if kbd_data is None or kbd_data.empty:
                        logger.warning("Не удалось загрузить актуальные данные КБД, пробуем использовать локальные")
                        kbd_data = downloader.load_kbd_data()
                        print(f"Загружены сохраненные данные КБД: {len(kbd_data) if kbd_data is not None else 0} записей")
                else:
                    # Загружаем сохраненные данные
                    logger.info("Загружаем локальные данные КБД")
                    kbd_data = downloader.load_kbd_data()
                    print(f"Загружены сохраненные данные КБД: {len(kbd_data) if kbd_data is not None else 0} записей")
            
            if kbd_data is not None:
                logger.info(f"Данные КБД доступны: {len(kbd_data)} записей")
                
                # Сохраняем копию используемых данных КБД в директорию results
                result_kbd_path = os.path.join(results_dir, f'kbd_data_{run_id}.csv')
                kbd_data.to_csv(result_kbd_path, index=False)
                logger.info(f"Копия данных КБД сохранена в {result_kbd_path}")
                
                # Шаг 2: Анализ КБД и получение рекомендаций
                from pys.data_collection.kbd_analyzer import KBDAnalyzer
                
                kbd_analyzer = KBDAnalyzer(kbd_data=kbd_data, output_dir=kbd_dir)
                
                # Визуализация КБД
                kbd_chart_path = kbd_analyzer.visualize_kbd(
                    save_path=os.path.join(viz_dir, f'kbd_curve_{run_id}.png')
                )
                
                # Получение комплексных рекомендаций для выбора облигаций
                comprehensive_recommendations = kbd_analyzer.get_comprehensive_recommendations()
                
                # Выделяем параметры фильтрации и стратегию взвешивания
                filter_params = {k: v for k, v in comprehensive_recommendations.items() 
                                if k in ['min_yield', 'max_yield', 'min_duration', 'max_duration', 
                                         'market_condition', 'excluded_issuers']}
                                        
                # Сохраняем рекомендуемую стратегию взвешивания
                recommended_weighting = comprehensive_recommendations.get('weighting_strategy')
                weighting_reason = comprehensive_recommendations.get('strategy_reason')
                
                # Применяем корректировки на основе выбранной стратегии
                if 'min_yield' in filter_params:
                    # Корректируем минимальную доходность
                    adjusted_min_yield = filter_params['min_yield'] + kbd_yield_adjustment
                    filter_params['min_yield'] = max(3.0, adjusted_min_yield)  # Не менее 3%
                    
                    # Пропорционально корректируем максимальную доходность
                    if 'max_yield' in filter_params:
                        yield_range = filter_params['max_yield'] - filter_params['min_yield']
                        filter_params['max_yield'] = filter_params['min_yield'] + yield_range * strategy_settings['yield_range_factor']
                
                # Корректируем дюрацию с учетом гибкости и профиля стратегии
                if 'min_duration' in filter_params and 'max_duration' in filter_params:
                    mid_duration = (filter_params['min_duration'] + filter_params['max_duration']) / 2
                    duration_range = filter_params['max_duration'] - filter_params['min_duration']
                    
                    # Увеличиваем диапазон дюрации для большей гибкости
                    filter_params['min_duration'] = max(0.1, mid_duration - (duration_range * kbd_duration_flexibility / 2))
                    filter_params['max_duration'] = mid_duration + (duration_range * kbd_duration_flexibility / 2)
                    
                    # Дополнительно корректируем в зависимости от профиля стратегии
                    filter_params['min_duration'] *= strategy_settings['duration_min_factor']
                    filter_params['max_duration'] *= strategy_settings['duration_max_factor']
                
                kbd_results = {
                    'kbd_metrics': kbd_analyzer.get_latest_kbd_metrics(),
                    'kbd_chart_path': kbd_chart_path,
                    'recommendations': comprehensive_recommendations,
                    'adjusted_recommendations': filter_params.copy(),  # Копия для отслеживания изменений
                    'weighting_recommendation': {
                        'strategy': recommended_weighting,
                        'reason': weighting_reason,
                        'description': comprehensive_recommendations.get('strategy_description', '')
                    }
                }
                
                # Сохраняем результаты анализа КБД в директорию analysis
                analysis_path = os.path.join(analysis_dir, f'kbd_analysis_{run_id}.json')
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(
                        {
                            'kbd_metrics': {k: str(v) if isinstance(v, (pd.Timestamp, datetime)) else v 
                                           for k, v in kbd_results['kbd_metrics'].items()},
                            'recommendations': kbd_results['recommendations'],
                            'adjusted_recommendations': kbd_results['adjusted_recommendations'],
                            'weighting_recommendation': kbd_results['weighting_recommendation']
                        }, 
                        f, 
                        indent=2
                    )
                logger.info(f"Результаты анализа КБД сохранены в {analysis_path}")
                
                logger.info(f"КБД анализ выполнен успешно. Рекомендуемая стратегия взвешивания: {recommended_weighting} ({weighting_reason})")
                print(f"КБД визуализация сохранена в: {kbd_chart_path}")
                print(f"Рекомендуемая стратегия взвешивания: {recommended_weighting} - {comprehensive_recommendations.get('strategy_description', '')}")
            else:
                logger.warning("Данные КБД недоступны")
                filter_params = _get_default_filter_params(strategy_profile)
                print("Внимание: Данные КБД недоступны, используются стандартные параметры")
                
        except Exception as e:
            logger.error(f"Ошибка при анализе КБД: {e}")
            logger.error(traceback.format_exc())
            filter_params = _get_default_filter_params(strategy_profile)
            print("Внимание: Ошибка при анализе КБД, используются стандартные параметры")
    else:
        # Если не требуется использование КБД
        filter_params = _get_default_filter_params(strategy_profile)
        logger.info("КБД анализ пропущен, используются стандартные параметры")
        print("Внимание: Используются стандартные параметры фильтрации")
    
    # Если стратегия взвешивания не указана явно, используем рекомендованную КБД
    if weighting_strategy is None:
        if recommended_weighting:
            weighting_strategy = recommended_weighting
            logger.info(f"Используется рекомендованная КБД стратегия взвешивания: {weighting_strategy}")
        else:
            # Значение по умолчанию, если нет рекомендаций
            weighting_strategy = 'inverse_duration'
            logger.info(f"Используется стандартная стратегия взвешивания: {weighting_strategy}")
    
    # Добавляем исключаемые эмитенты из параметров
    if excluded_issuers:
        filter_params['excluded_issuers'] = excluded_issuers
        
    # Перезаписываем параметры, если заданы вручную
    if override_params:
        filter_params.update(override_params)
        logger.info(f"Параметры переопределены вручную: {override_params}")
        print(f"Внимание: Используются модифицированные параметры фильтрации")
    
    print("\nПараметры фильтрации облигаций:")
    for key, value in filter_params.items():
        print(f"  {key}: {value}")
    
    # Итеративный процесс формирования портфеля с адаптивной корректировкой параметров
    bond_results = None
    current_params = filter_params.copy()
    
    for iteration in range(max_adjustment_iterations):
        # Шаг 3: Запуск выбора облигаций с текущими параметрами
        bond_results = run_pipeline_bonds_processor(
            base_path=base_path,
            dataset_path=dataset_path,
            results_dir=portfolios_dir,  # Сохраняем портфели в отдельной папке
            min_yield=current_params.get('min_yield', 6.0),
            max_yield=current_params.get('max_yield', 22.0),
            min_duration=current_params.get('min_duration', 0.1),
            max_duration=current_params.get('max_duration', 5.0),
            excluded_issuers=current_params.get('excluded_issuers', None),
            n_bonds=n_bonds,
            weighting_strategy=weighting_strategy,  # Используем рекомендованную или указанную стратегию
            portfolio_stability=portfolio_stability,
        )
        
        # Проверяем, получили ли мы достаточное количество облигаций
        if (bond_results.get('success', False) and 
            bond_results.get('portfolio') is not None and 
            len(bond_results['portfolio']) >= min_bonds):
            logger.info(f"Сформирован портфель из {len(bond_results['portfolio'])} облигаций (достаточно)")
            break
        else:
            # Если портфель не сформирован или содержит слишком мало облигаций, корректируем параметры
            portfolio_size = len(bond_results.get('portfolio', [])) if bond_results.get('portfolio') is not None else 0
            logger.warning(f"Итерация {iteration+1}: Сформировано только {portfolio_size} облигаций (минимум {min_bonds})")
            
            # Корректируем параметры для следующей итерации
            if 'min_yield' in current_params:
                current_params['min_yield'] = max(3.0, current_params['min_yield'] - 2.0)  # Снижаем на 2 п.п.
                
            if 'max_duration' in current_params:
                current_params['max_duration'] = min(10.0, current_params['max_duration'] + 1.0)  # Увеличиваем на 1 год
                
            if 'min_duration' in current_params:
                current_params['min_duration'] = max(0.1, current_params['min_duration'] - 0.1)  # Уменьшаем на 0.1 года
                
            logger.info(f"Скорректированные параметры для итерации {iteration+2}: {current_params}")
            print(f"\nНедостаточная диверсификация. Корректировка параметров (итерация {iteration+1}):")
            print(f"  Новая мин. доходность: {current_params.get('min_yield', 'N/A')}")
            print(f"  Новый диапазон дюрации: {current_params.get('min_duration', 'N/A')}-{current_params.get('max_duration', 'N/A')}")
    
    # Если после всех итераций не удалось сформировать достаточный портфель
    if bond_results is None or not bond_results.get('success', False) or bond_results.get('portfolio') is None:
        logger.error("Не удалось сформировать портфель облигаций")
        print("Не удалось сформировать портфель облигаций")
        return {'success': False, 'error': 'Не удалось сформировать портфель облигаций'}
    
    # Записываем итоговые использованные параметры
    final_params = current_params.copy()
    print(f"\nИтоговые параметры фильтрации:")
    for key, value in final_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем информацию о корректировке параметров
    if kbd_results:
        kbd_results['final_recommendations'] = final_params
    
    # Сохраняем результаты в директорию results
    results_summary = {
        'run_id': run_id,
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategy_profile': strategy_profile,
        'weighting_strategy_used': weighting_strategy,
        'initial_params': filter_params,
        'final_params': final_params,
        'portfolio_stats': bond_results.get('stats', {})
    }
    
    # Сохраняем результаты в JSON
    results_path = os.path.join(results_dir, f'bonds_results_{run_id}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(results_summary, f, indent=2)
    logger.info(f"Сводка результатов сохранена в {results_path}")
    
    # Шаг 4: Создание отчета с объединенными результатами
    if bond_results.get('success', False):
        report_path = create_combined_report(bond_results, kbd_results, reports_dir, run_id)
        # Добавляем путь к отчету в результаты
        if report_path:
            bond_results['report_path'] = report_path
    
    # Шаг 5: Возвращаем объединенные результаты (в соответствии с output_format)
    
    # Базовые результаты всегда включены
    results = {
        'success': bond_results.get('success', False),
        'run_id': run_id
    }
    
    # Добавляем данные в соответствии с выбранным форматом вывода
    if output_format == 'all':
        results.update({
            'bond_results': bond_results,
            'kbd_results': kbd_results,
            'initial_params': filter_params,
            'final_params': final_params,
            'weighting_strategy': weighting_strategy,
            'portfolio': bond_results.get('portfolio'),
            'stats': bond_results.get('stats'),
            'paths': {
                'report': bond_results.get('report_path'),
                'kbd_chart': kbd_results.get('kbd_chart_path'),
                'portfolio_viz': bond_results.get('visualization_path'),
                'results_summary': results_path,
                'portfolio_path' : bond_results.get('portfolio_path')
            }
        })
    elif output_format == 'portfolio':
        results.update({
            'portfolio': bond_results.get('portfolio'),
            'weighting_strategy': weighting_strategy
        })
    elif output_format == 'metrics':
        results.update({
            'stats': bond_results.get('stats'),
            'parameters': final_params,
            'strategy': weighting_strategy
        })
    elif output_format == 'viz':
        results.update({
            'paths': {
                'report': bond_results.get('report_path'),
                'kbd_chart': kbd_results.get('kbd_chart_path'),
                'portfolio_viz': bond_results.get('visualization_path')
            }
        })
    
    logger.info("Пайплайн выбора облигаций с учетом КБД завершен")
    return results


def _get_strategy_profile_settings(profile):
    """
    Получить настройки для выбранного профиля стратегии
    
    Args:
        profile: профиль стратегии ('conservative', 'balanced', 'aggressive')
    
    Returns:
        dict: настройки стратегии
    """
    profiles = {
        'conservative': {
            'yield_range_factor': 1.2,   # Более узкий диапазон доходности
            'duration_min_factor': 1.0,  # Стандартный нижний предел дюрации
            'duration_max_factor': 0.8,  # Сниженный верхний предел дюрации
            'min_yield_adjustment': -1,  # Снижение требуемой доходности
            'yield_threshold_factor': 0.9  # Меньшая требуемая доходность
        },
        'balanced': {
            'yield_range_factor': 1.5,
            'duration_min_factor': 0.9,
            'duration_max_factor': 1.0,
            'min_yield_adjustment': 0,
            'yield_threshold_factor': 1.0
        },
        'aggressive': {
            'yield_range_factor': 2.0,    # Более широкий диапазон доходности
            'duration_min_factor': 0.8,   # Допускаем меньшую дюрацию
            'duration_max_factor': 1.2,   # Допускаем большую дюрацию
            'min_yield_adjustment': 1,    # Повышение требуемой доходности
            'yield_threshold_factor': 1.1  # Большая требуемая доходность
        }
    }
    
    # Возвращаем настройки для выбранного профиля или для сбалансированного по умолчанию
    return profiles.get(profile, profiles['balanced'])

def _get_default_filter_params(strategy_profile):
    """
    Получить стандартные параметры фильтрации в зависимости от профиля стратегии
    
    Args:
        strategy_profile: профиль стратегии
        
    Returns:
        dict: параметры фильтрации
    """
    base_params = {
        'min_yield': 6.0,
        'max_yield': 22.0,
        'min_duration': 0.1,
        'max_duration': 5.0,
        'market_condition': 'unknown'
    }
    
    settings = _get_strategy_profile_settings(strategy_profile)
    
    # Корректируем параметры в зависимости от профиля
    if strategy_profile == 'conservative':
        base_params.update({
            'min_yield': 5.0,
            'max_yield': 18.0,
            'min_duration': 0.1,
            'max_duration': 3.0
        })
    elif strategy_profile == 'aggressive':
        base_params.update({
            'min_yield': 7.0,
            'max_yield': 25.0,
            'min_duration': 0.1,
            'max_duration': 7.0
        })
    
    return base_params

def create_combined_report(bond_results, kbd_results, reports_dir, run_id=None):
    """
    Создать комбинированный отчет по результатам анализа с информацией об адаптации параметров
    
    Args:
        bond_results: результаты выбора облигаций
        kbd_results: результаты анализа КБД
        reports_dir: директория для сохранения отчета
        run_id: идентификатор запуска
        
    Returns:
        str: путь к созданному отчету
    """
    logger = BaseLogger('BondsKBDReport').logger
    
    try:
        # Создаем идентификатор запуска, если не предоставлен
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # Создаем итоговый отчет в HTML
        report_path = os.path.join(reports_dir, f'bonds_kbd_report_{run_id}.html')
        
        # Получаем данные для отчета
        portfolio = bond_results.get('portfolio')
        stats = bond_results.get('stats', {})
        kbd_metrics = kbd_results.get('kbd_metrics', {})
        
        # Получаем информацию об адаптации параметров
        initial_recommendations = kbd_results.get('recommendations', {})
        final_recommendations = kbd_results.get('final_recommendations', {})
        
        if portfolio is None:
            logger.warning("Нет данных портфеля для отчета")
            return None
        
        # Генерируем HTML
        html_content = f"""
        <html>
        <head>
            <title>Отчет по выбору облигаций с учетом КБД</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .highlight {{ background-color: #e8f4f8; }}
                .changes {{ background-color: #fff8e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .charts {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .chart {{ margin-bottom: 20px; box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); }}
            </style>
        </head>
        <body>
            <h1>Отчет по выбору облигаций с учетом КБД</h1>
            <p>Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
            <p>ID запуска: {run_id}</p>
            
            <h2>Текущие рыночные условия</h2>
            <div class="stats">
                <p><strong>Состояние рынка:</strong> {kbd_metrics.get('market_state', 'неизвестно')}</p>
                <p><strong>Дата КБД:</strong> {kbd_metrics.get('date', 'неизвестно')}</p>
                <p><strong>Ключевые точки КБД:</strong></p>
                <ul>
        """
        
        # Добавляем значения КБД
        kbd_values = kbd_metrics.get('kbd_values', {})
        for tenor, value in kbd_values.items():
            html_content += f"<li>Доходность на {tenor}: {value:.2f}%</li>"
        
        html_content += f"""
                </ul>
            </div>
            
            <h2>Параметры выбора облигаций</h2>
            <div class="stats">
                <p><strong>Целевая доходность:</strong> {kbd_metrics.get('yield_threshold', 'не указано')}%</p>
                <p><strong>Оптимальная дюрация:</strong> {kbd_metrics.get('optimal_duration', (0, 0))[0]} - {kbd_metrics.get('optimal_duration', (0, 0))[1]} лет</p>
            </div>
        """
        
        # Добавляем раздел о стратегии взвешивания
        weighting_info = kbd_results.get('weighting_recommendation', {})
        if weighting_info:
            html_content += f"""
            <h3>Стратегия взвешивания облигаций</h3>
            <div class="stats">
                <p><strong>Рекомендованная стратегия:</strong> {weighting_info.get('strategy', 'не определена')}</p>
                <p><strong>Обоснование:</strong> {weighting_info.get('description', '')}</p>
            </div>
            """
        
        # Добавляем раздел с информацией об адаптации параметров
        if initial_recommendations != final_recommendations:
            html_content += """
            <h3>Адаптация параметров для обеспечения диверсификации</h3>
            <div class="changes">
                <p>Для обеспечения достаточной диверсификации портфеля были адаптированы следующие параметры:</p>
                <table>
                    <tr>
                        <th>Параметр</th>
                        <th>Начальное значение</th>
                        <th>Итоговое значение</th>
                    </tr>
            """
            
            # Добавляем строки с изменениями параметров
            params_to_check = ['min_yield', 'max_yield', 'min_duration', 'max_duration']
            for param in params_to_check:
                if param in initial_recommendations and param in final_recommendations:
                    initial_value = initial_recommendations[param]
                    final_value = final_recommendations[param]
                    
                    # Форматируем значения
                    if isinstance(initial_value, (int, float)) and isinstance(final_value, (int, float)):
                        initial_value = f"{initial_value:.2f}"
                        final_value = f"{final_value:.2f}"
                    
                    html_content += f"""
                    <tr>
                        <td>{param}</td>
                        <td>{initial_value}</td>
                        <td>{final_value}</td>
                    </tr>
                    """
            
            html_content += """
                </table>
            </div>
            """
            
        html_content += f"""
            <h2>Выбранный портфель облигаций</h2>
            <table>
                <tr>
                    <th>Код</th>
                    <th>Название</th>
                    <th>Доходность, %</th>
                    <th>Дюрация, лет</th>
                    <th>Вес, %</th>
                </tr>
        """
        
        # Добавляем строки портфеля
        for _, bond in portfolio.iterrows():
            html_content += "<tr>"
            html_content += f"<td>{bond.get('security_code', '')}</td>"
            html_content += f"<td>{bond.get('full_name', '')}</td>"
            html_content += f"<td>{bond.get('yield', 0):.2f}</td>"
            html_content += f"<td>{bond.get('duration_years', 0):.2f}</td>"
            html_content += f"<td>{bond.get('weight', 0)*100:.2f}</td>"
            html_content += "</tr>"
        
        html_content += f"""
            </table>
            
            <h2>Характеристики портфеля</h2>
            <div class="stats">
                <p><strong>Средневзвешенная доходность:</strong> {stats.get('weighted_yield', 0):.2f}%</p>
                <p><strong>Средневзвешенная дюрация:</strong> {stats.get('weighted_duration', 0):.2f} лет</p>
                <p><strong>Количество облигаций:</strong> {stats.get('number_of_bonds', 0)}</p>
            </div>
            
            <h2>Графики</h2>
            <div class="charts">
        """
        
        # Добавляем ссылки на графики
        if kbd_results.get('kbd_chart_path'):
            chart_rel_path = os.path.relpath(kbd_results['kbd_chart_path'], reports_dir)
            html_content += f"""
                <div class="chart">
                    <h3>Кривая бескупонной доходности</h3>
                    <img src="../viz/{os.path.basename(kbd_results['kbd_chart_path'])}" alt="Кривая бескупонной доходности" style="max-width: 100%;">
                </div>
            """
        
        if bond_results.get('visualization_path'):
            chart_rel_path = os.path.relpath(bond_results['visualization_path'], reports_dir)
            html_content += f"""
                <div class="chart">
                    <h3>Карта облигаций (доходность/дюрация)</h3>
                    <img src="../portfolios/{os.path.basename(bond_results['visualization_path'])}" alt="Карта облигаций" style="max-width: 100%;">
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Записываем HTML в файл
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Комбинированный отчет сохранен в {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Ошибка при создании отчета: {e}")
        logger.error(traceback.format_exc())
        return None
