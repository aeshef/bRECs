# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-27 20:44:58
**Идентификатор запуска:** run_20250427_204437_balanced_portfolio_0427_2044
**Профиль стратегии:** aggresive

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

## Результаты

### Стандартный портфель (Markowitz)

- Ожидаемая доходность: 2.38%
- Ожидаемая волатильность: 12.09%
- Коэффициент Шарпа: -0.42

### ЛУЧШИЙ ПОРТФЕЛЬ

**Тип портфеля: STANDARD**

- Ожидаемая доходность: 2.38%
- Ожидаемая волатильность: 12.09%
- Коэффициент Шарпа: -0.42

## Веса в итоговом портфеле

| Актив | Вес |
|-------|-----|
| RISK_FREE | 40.00% |
| GAZP | 9.00% |
| AFKS | 9.00% |
| RUAL | 9.00% |
| TATN | 9.00% |
| SBER | 9.00% |
| LKOH | 9.00% |
| MTSS | 6.00% |

## Графики

Визуализации доступны в директории финального портфеля:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_204437_balanced_portfolio_0427_2044/final_portfolio`

## Расположение результатов

Все результаты этого запуска сохранены в директории:
`/Users/aeshef/Documents/GitHub/kursach/data/pipeline_runs/run_20250427_204437_balanced_portfolio_0427_2044`

### Структура директорий

```
run_20250427_204437_balanced_portfolio_0427_2044/
├── signals/            # Сигналы для акций
├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)
├── shorts_portfolio/   # Портфель с короткими позициями
├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)
├── backtest/           # Результаты бэктестов
├── final_portfolio/    # Лучший выбранный портфель
└── bond_portfolio.csv  # Портфель облигаций
```
