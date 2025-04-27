# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-27 20:57:24
**Идентификатор запуска:** run_20250427_205618_balanced_portfolio_0427_2056
**Профиль стратегии:** aggressive

## Параметры

### Параметры генерации сигналов
- weight_tech: 0.5
- weight_sentiment: 0.3
- weight_fundamental: 0.2
- threshold_buy: 0.5
- threshold_sell: -0.5
- top_pct: 0.3

### Параметры оптимизации стандартного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- include_short_selling: False

### Параметры портфеля с короткими позициями
- risk_free_rate: 0.075
- train_period: ('2024-01-01', '2024-12-31')
- test_period: ('2025-01-01', '2025-04-15')
- include_short_selling: True
- verify_with_honest_backtest: True

### Параметры комбинированного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- long_ratio: 0.7
- include_short_selling: True

## Результаты

### Стандартный портфель (Markowitz)

- Ожидаемая доходность: 2.38%
- Ожидаемая волатильность: 12.09%
- Коэффициент Шарпа: -0.42

### Портфель с короткими позициями

- Годовая доходность: -1.85%
- Коэффициент Шарпа: -2.12
- Максимальная просадка: -6.42%

### Комбинированный портфель

- Ожидаемая доходность: 4.83%
- Ожидаемая волатильность: 10.38%
- Коэффициент Шарпа: -0.26

### ЛУЧШИЙ ПОРТФЕЛЬ

**Тип портфеля: SHORT**

- Ожидаемая доходность: -1.85%
- Ожидаемая волатильность: 4.40%
- Коэффициент Шарпа: -2.12

## Веса в итоговом портфеле

| Актив | Вес |
|-------|-----|
| RISK_FREE | 30.00% |
| AFKS | 14.00% |
| RUAL | 14.00% |
| SBER | 14.00% |
| TATN | 14.00% |
| LKOH | 14.00% |

## Графики

Визуализации доступны в директории финального портфеля:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_205618_balanced_portfolio_0427_2056/final_portfolio`

## Расположение результатов

Все результаты этого запуска сохранены в директории:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_205618_balanced_portfolio_0427_2056`

### Структура директорий

```
run_20250427_205618_balanced_portfolio_0427_2056/
├── signals/            # Сигналы для акций
├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)
├── shorts_portfolio/   # Портфель с короткими позициями
├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)
├── backtest/           # Результаты бэктестов
├── final_portfolio/    # Лучший выбранный портфель
└── bond_portfolio.csv  # Портфель облигаций
```
