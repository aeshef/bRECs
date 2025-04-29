from datetime import datetime
from pathlib import Path

# --- Пути ---
PROJECT_ROOT = Path("/Users/aeshef/Desktop/FOR3.9TEST/kursach")
PYS_PATH = PROJECT_ROOT / "pys"
DATA_PATH = PROJECT_ROOT / "data"
KBD_DATA_PATH = DATA_PATH / "processed_data/BONDS/kbd/data/kbd_data.csv"

# --- Общие Параметры ---
COMMON_PARAMS = {
    "start_date": "2024-01-01",
    "end_date": "2025-04-15", # Используется для KBD и, возможно, других пайплайнов
   'train_period': ('2024-01-01', '2024-12-31'),
    'test_period': ('2025-01-01', '2025-04-15'),
    "risk_free_rate": 0.075, # Годовая безрисковая ставка
    "min_rf_allocation": 0.3, # Минимальная доля облигаций (для умеренной стратегии)
    "max_rf_allocation": 0.5, # Максимальная доля облигаций (для умеренной стратегии)
    "max_weight": 0.15       # Максимальный вес одной акции (для умеренной стратегии)
}

# --- Параметры KBD ---
KBD_PARAMS = {
    "min_bonds": 5,               # Минимальное кол-во облигаций для портфеля
    "max_threshold": 99,          # Максимальный порог непрерывности в данных MOEX
    "kbd_yield_adjustment": -2.0, # Корректировка доходности KBD
    "portfolio_stability": 0.7,   # Стабильность портфеля при выборе облигаций
    "kbd_duration_flexibility": 1.5, # Гибкость дюрации
    "max_adjustment_iterations": 3  # Макс. итераций настройки весов облигаций
}

# --- Параметры Сигналов ---
SIGNAL_PARAMS = {
    "weight_tech": 0.5,
    "weight_sentiment": 0.3,
    "weight_fundamental": 0.2,
    "threshold_buy": 0.5,
    "threshold_sell": -0.5,
    "top_pct": 0.3, # Процент лучших акций для shortlist
    "tech_indicators": [
        'RSI_14', 'MACD_diff', 'Stoch_%K',
        'CCI_20', 'Williams_%R_14', 'ROC_10'
    ],
    "sentiment_indicators": [
        'sentiment_compound_median', 'sentiment_direction',
        'sentiment_ma_7d', 'sentiment_ratio', 'sentiment_zscore_7d'
    ],
    "fund_weights": { # Веса для фундаментальных метрик
        "Чистая прибыль, млрд руб": 0.10, "Див доход, ао, %": 0.10,
        "Дивиденды/прибыль, %": 0.05, "EBITDA, млрд руб": 0.08,
        "FCF, млрд руб": 0.10, "Рентаб EBITDA, %": 0.08,
        "Чистый долг, млрд руб": 0.08, "Долг/EBITDA": 0.07,
        "EPS, руб": 0.07, "ROE, %": 0.10, "ROA, %": 0.08, "P/E": 0.09
    }
}

# --- Параметры Сбора Данных ---
MARKET_DATA_PARAMS = {
    "timeframe": "1h" # Таймфрейм для цен акций
}

NEWS_PARAMS = {
    "telegram_channel": "cbrstocks",  # Канал Telegram для новостей
    "telegram_limit": 1000,         # Лимит сообщений для загрузки
    "max_history_days": 90,         # Макс. глубина истории новостей
    "use_cached_telegram": False,   # Использовать ли кэш Telethon
    "cleanup_old_files": True,      # Удалять ли старые файлы новостей
}

# --- Параметры интеграции котировок с сигналами (Глобальные и по Профилям) ---
INTEGRATION_PARAMS = {
    "method": "zero" # Метод заполнения пропусков при интеграции
}

# --- Параметры Портфеля (Глобальные и по Профилям) ---
# Общие параметры для всех типов портфелей
PORTFOLIO_BASE_PARAMS = {
    "min_position_weight": 0.01, # Минимальный вес любой позиции
    "verify_with_honest_backtest": False, # Не запускать бэктест при генерации
    'max_assets' : 15,
    'min_assets' : 5
}

# Параметры, зависящие от риск-профиля
PORTFOLIO_PARAMS = {
    "conservative": {
        **PORTFOLIO_BASE_PARAMS,
        "min_rf_allocation": 0.5,
        "max_rf_allocation": 0.7,
        "max_weight": 0.10,
        "allow_short": False, # Консервативные не шортят
        "long_ratio": 1.0    # Только лонг
    },
    "moderate": {
        **PORTFOLIO_BASE_PARAMS,
        "min_rf_allocation": COMMON_PARAMS['min_rf_allocation'], # Из общих
        "max_rf_allocation": COMMON_PARAMS['max_rf_allocation'], # Из общих
        "max_weight": COMMON_PARAMS['max_weight'],              # Из общих
        "allow_short": True, # Умеренные не шортят (можно изменить)
        "long_ratio": 1.0
    },
    "aggressive": {
        **PORTFOLIO_BASE_PARAMS,
        "min_rf_allocation": 0.1, # Меньше облигаций
        "max_rf_allocation": 0.3,
        "max_weight": 0.20,       # Больший вес одной акции
        "allow_short": True,      # Агрессивные могут шортить
        "long_ratio": 0.7         # Соотношение лонг/шорт
    }
}

# Добавим параметры для конкретных типов портфелей (standard, short, combined)
# Они будут объединены с параметрами профиля в pipeline_runner
PORTFOLIO_TYPE_PARAMS = {
    "standard": {
        "include_short_selling": False
    },
    "short": {
        # verify_with_honest_backtest уже в PORTFOLIO_BASE_PARAMS
        "best_params_file": None # Только для бэктеста с Grid Search
    },
    "combined": {
        "include_short_selling": True
        # long_ratio уже определяется из профиля
    }
}

# --- Параметры Оптимизации ---
OPTIMIZATION_PARAMS = {
    "tau": 0.05,              # Параметр Black-Litterman
    "default_optimization": "markowitz", # Метод по умолчанию
    "views": None,            # Сюда можно передавать прогнозы
    "view_confidences": None,
    "market_caps": None       # Сюда можно передать капитализации
}

# --- Параметры Бэктеста (Отключены при генерации) ---
BACKTEST_PARAMS = {
    "train_period": COMMON_PARAMS['train_period'], # Используем общие
    "test_period": COMMON_PARAMS['test_period'],   # Используем общие
    "risk_free_rate": COMMON_PARAMS['risk_free_rate'],
    "use_grid_search_params": False # Не используем при генерации
}

# --- Параметры Выбора Лучшего Портфеля (по Профилям) ---
SELECT_PARAMS_BY_PROFILE = {
    "conservative": {
        "metrics_priority": ['sharpe', 'return', 'volatility'],
        "min_sharpe": -1.0, # Требование к Шарпу для консервативного
        "prefer_standard": True, # Всегда выбираем стандартный
        "force_portfolio_type": "standard" # Форсируем стандартный
    },
    "moderate": {
        "metrics_priority": ['sharpe', 'return', 'volatility'],
        "min_sharpe": -1.0, # Меньшее требование к Шарпу
        "prefer_standard": True, # Предпочитаем стандартный
        "force_portfolio_type": None # Не форсируем
    },
    "aggressive": {
        "metrics_priority": ['return', 'sharpe', 'volatility'],
        "min_sharpe": -1.0,
        "prefer_standard": False,
        "force_portfolio_type": None
    }
}

# --- Контроль Запуска Разных Типов Портфелей ---
# Этот словарь будет использоваться в pipeline_runner для определения,
# какие типы портфелей запускать, с учетом preferences.allow_short
PORTFOLIO_CONTROLS = {
    'run_standard_portfolio': True,
    'run_short_portfolio': True, # Будет проверен allow_short и профиль в pipeline_runner
    'run_combined_portfolio': True, # Будет проверен allow_short и профиль в pipeline_runner
    'override_risk_profile': False # Не переопределяем профиль пользователя
}


# --- Параметры Отчетов и Визуализации (Отключены для бота) ---
REPORT_PARAMS = {
    'include_charts': False,
    'include_metrics': False,
    'include_weights': False,
    'report_format': None # Не генерируем отчеты
}

VISUALIZATION_PARAMS = {
    'plot_style': 'ggplot',
    'chart_size': (12, 8),
    'dpi': 300
}

TICKERS = [
    'SBER', 'GAZP', 'LKOH', 'GMKN', 'ROSN',
    'TATN', 'MTSS', 'ALRS', 'SNGS', 'NVTK',
    'MVID', 'PHOR', 'SIBN', 'AFKS', 'MAGN', 'RUAL','AFLT',
    'CBOM',
    'POSI',
    'PLZL'
]
