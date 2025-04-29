# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-29 18:03:42
**Идентификатор запуска:** run_20250429_180023_balanced_portfolio_0429_1751
**Профиль стратегии:** aggressive

## Параметры

### Параметры генерации сигналов
- weight_tech: 0.6
- weight_sentiment: 0.3
- weight_fundamental: 0.1
- threshold_buy: 0.4
- threshold_sell: -0.4
- top_pct: 0.3
- tech_indicators: ['RSI_14', 'MACD_diff', 'Stoch_%K', 'CCI_20', 'Williams_%R_14', 'ROC_10']
- sentiment_indicators: ['sentiment_compound_median', 'sentiment_direction', 'sentiment_ma_7d', 'sentiment_ratio', 'sentiment_zscore_7d']
- fund_weights: {'Чистая прибыль, млрд руб': 0.1, 'Див доход, ао, %': 0.1, 'Дивиденды/прибыль, %': 0.05, 'EBITDA, млрд руб': 0.08, 'FCF, млрд руб': 0.1, 'Рентаб EBITDA, %': 0.08, 'Чистый долг, млрд руб': 0.08, 'Долг/EBITDA': 0.07, 'EPS, руб': 0.07, 'ROE, %': 0.1, 'ROA, %': 0.08, 'P/E': 0.09}

### Параметры оптимизации стандартного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.4
- max_rf_allocation: 0.6
- max_weight: 0.4
- include_short_selling: False

### Параметры портфеля с короткими позициями
- risk_free_rate: 0.075
- train_period: ('2024-01-01', '2024-12-31')
- test_period: ('2025-01-01', '2025-04-15')
- include_short_selling: True
- verify_with_honest_backtest: True

### Параметры комбинированного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.4
- max_rf_allocation: 0.6
- max_weight: 0.4
- long_ratio: 0.7
- include_short_selling: True

## Результаты

### Стандартный портфель (Markowitz)

- Ожидаемая доходность: 19.89%
- Ожидаемая волатильность: 10.28%
- Коэффициент Шарпа: 1.20

### Портфель с короткими позициями

- Годовая доходность: -3.92%
- Коэффициент Шарпа: -1.78
- Максимальная просадка: -8.92%

### Комбинированный портфель

- Ожидаемая доходность: 14.16%
- Ожидаемая волатильность: 8.46%
- Коэффициент Шарпа: 0.79

### ЛУЧШИЙ ПОРТФЕЛЬ

**Тип портфеля: COMBINED**

- Ожидаемая доходность: 14.16%
- Ожидаемая волатильность: 8.46%
- Коэффициент Шарпа: 0.79

## Веса в итоговом портфеле

| Актив | Вес |
|-------|-----|
| RISK_FREE | 50.00% |
| LKOH | 16.67% |
| PLZL | 16.67% |
| SBER | 16.67% |

## Графики

Визуализации доступны в директории финального портфеля:
`/Users/aeshef/Desktop/FOR3.9TEST/kursach/data/pipeline_runs/run_20250429_180023_balanced_portfolio_0429_1751/final_portfolio`

## Расположение результатов

Все результаты этого запуска сохранены в директории:
`/Users/aeshef/Desktop/FOR3.9TEST/kursach/data/pipeline_runs/run_20250429_180023_balanced_portfolio_0429_1751`

### Структура директорий

```
run_20250429_180023_balanced_portfolio_0429_1751/
├── signals/            # Сигналы для акций
├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)
├── shorts_portfolio/   # Портфель с короткими позициями
├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)
├── backtest/           # Результаты бэктестов
├── final_portfolio/    # Лучший выбранный портфель
└── bond_portfolio.csv  # Портфель облигаций
```
