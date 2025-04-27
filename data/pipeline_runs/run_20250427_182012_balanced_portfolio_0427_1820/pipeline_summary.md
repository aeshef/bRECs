# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-27 18:20:28
**Идентификатор запуска:** run_20250427_182012_balanced_portfolio_0427_1820
**Профиль стратегии:** moderate

## Параметры

### Параметры генерации сигналов
- weight_tech: 0.5
- weight_sentiment: 0.3
- weight_fundamental: 0.2
- threshold_buy: 0.5
- threshold_sell: -0.5
- top_pct: 0.3

### Параметры оптимизации стандартного портфеля
- risk_free_rate: 0.1
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- include_short_selling: False
- optimization: markowitz

### Параметры комбинированного портфеля
- risk_free_rate: 0.1
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- long_ratio: 0.7
- include_short_selling: True

## Результаты

### Комбинированный портфель

- Ожидаемая доходность: 5.83%
- Ожидаемая волатильность: 10.38%
- Коэффициент Шарпа: -0.40

### ЛУЧШИЙ ПОРТФЕЛЬ

**Тип портфеля: COMBINED**

- Ожидаемая доходность: 5.83%
- Ожидаемая волатильность: 10.38%
- Коэффициент Шарпа: -0.40

## Веса в итоговом портфеле

| Актив | Вес |
|-------|-----|
| RISK_FREE | 40.00% |
| SIBN | 4.15% |
| ALRS | 4.15% |
| MAGN | 4.15% |
| GMKN | 4.15% |
| AFKS | 3.82% |
| PHOR | 3.82% |
| GAZP | 3.82% |
| MVID | 3.82% |
| NVTK | 3.82% |
| MTSS | 3.82% |
| RUAL | 3.82% |
| SNGS | 3.82% |
| TATN | 3.82% |
| SBER | 3.82% |
| LKOH | 3.82% |
| ROSN | 1.38% |

## Графики

Визуализации доступны в директории финального портфеля:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_182012_balanced_portfolio_0427_1820/final_portfolio`

## Расположение результатов

Все результаты этого запуска сохранены в директории:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_182012_balanced_portfolio_0427_1820`

### Структура директорий

```
run_20250427_182012_balanced_portfolio_0427_1820/
├── signals/            # Сигналы для акций
├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)
├── shorts_portfolio/   # Портфель с короткими позициями
├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)
├── backtest/           # Результаты бэктестов
├── final_portfolio/    # Лучший выбранный портфель
└── bond_portfolio.csv  # Портфель облигаций
```
