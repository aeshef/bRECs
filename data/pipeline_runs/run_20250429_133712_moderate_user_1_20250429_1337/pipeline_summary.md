# Отчет о запуске пайплайна

**Дата и время запуска:** 2025-04-29 13:37:33
**Идентификатор запуска:** run_20250429_133712_moderate_user_1_20250429_1337
**Профиль стратегии:** moderate

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
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- include_short_selling: False

### Параметры комбинированного портфеля
- risk_free_rate: 0.075
- min_rf_allocation: 0.3
- max_rf_allocation: 0.5
- max_weight: 0.15
- long_ratio: 1.0
- include_short_selling: True

