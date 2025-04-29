# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-29 16:08:47
**Идентификатор запуска:** run_20250429_160827_aggressive_user_1_20250429_1608
**Профиль стратегии:** aggressive

## Параметры

### Параметры генерации сигналов
- weight_tech: 0.5
- weight_sentiment: 0.3
- weight_fundamental: 0.2
- threshold_buy: 0.5
- threshold_sell: -0.5
- top_pct: 0.3
- tech_indicators: ['RSI_14', 'MACD_diff', 'Stoch_%K', 'CCI_20', 'Williams_%R_14', 'ROC_10']
- sentiment_indicators: ['sentiment_compound_median', 'sentiment_direction', 'sentiment_ma_7d', 'sentiment_ratio', 'sentiment_zscore_7d']
- fund_weights: {'Чистая прибыль, млрд руб': 0.1, 'Див доход, ао, %': 0.1, 'Дивиденды/прибыль, %': 0.05, 'EBITDA, млрд руб': 0.08, 'FCF, млрд руб': 0.1, 'Рентаб EBITDA, %': 0.08, 'Чистый долг, млрд руб': 0.08, 'Долг/EBITDA': 0.07, 'EPS, руб': 0.07, 'ROE, %': 0.1, 'ROA, %': 0.08, 'P/E': 0.09}

### Параметры оптимизации стандартного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.1
- max_rf_allocation: 0.3
- max_weight: 0.2
- include_short_selling: False

## Результаты

### Стандартный портфель (Markowitz)

- Ожидаемая доходность: 6.84%
- Ожидаемая волатильность: 15.26%
- Коэффициент Шарпа: -0.04

### ЛУЧШИЙ ПОРТФЕЛЬ

**Тип портфеля: STANDARD**

- Ожидаемая доходность: 6.84%
- Ожидаемая волатильность: 15.26%
- Коэффициент Шарпа: -0.04

## Веса в итоговом портфеле

| Актив | Вес |
|-------|-----|
| RISK_FREE | 20.00% |
| AFKS | 16.00% |
| RUAL | 16.00% |
| TATN | 16.00% |
| SBER | 16.00% |
| PHOR | 16.00% |

## Графики

Визуализации доступны в директории финального портфеля:
`/Users/aeshef/Desktop/FOR3.9TEST/kursach/data/pipeline_runs/run_20250429_160827_aggressive_user_1_20250429_1608/final_portfolio`

## Расположение результатов

Все результаты этого запуска сохранены в директории:
`/Users/aeshef/Desktop/FOR3.9TEST/kursach/data/pipeline_runs/run_20250429_160827_aggressive_user_1_20250429_1608`

### Структура директорий

```
run_20250429_160827_aggressive_user_1_20250429_1608/
├── signals/            # Сигналы для акций
├── portfolio/          # Стандартный портфель (Markowitz/Black-Litterman)
├── shorts_portfolio/   # Портфель с короткими позициями
├── combined_portfolio/ # Комбинированный портфель (длинные и короткие позиции)
├── backtest/           # Результаты бэктестов
├── final_portfolio/    # Лучший выбранный портфель
└── bond_portfolio.csv  # Портфель облигаций
```
