![Логотип](https://i.ibb.co/pVLH0Ky/IMAGE-2025-04-28-15-06-10.jpg)

## 📑 Оглавление
- [Описание](#описание)
- [Документация](#документация)
- [Инструкция по запуску](#инструкция-по-запуску)
- [Требования](#требования)
- [Контакты](#контакты)
- [Демонстрация работы](#демонстрацияработы)

## 📝 Описание
Это интеллектуальная рекомендательная система для формирования и оптимизации инвестиционного портфеля. Система объединяет данные технического, фундаментального и новостного анализа, реализует расчёт торговых сигналов, оптимизацию с использованием моделей Марковица и Блэка–Литермана, а также проводит бэктестирование стратегий. Дополнительно разработан Telegram-бот, позволяющий пользователю получить персонализированные рекомендации на основе оценки риск-профиля.

## 📚 Документация
Подробная документация проекта доступна по ссылке:  
👉 https://aeshef.github.io/kursach/index.html

## 🎥 Демонстрация работы

![Демонстрация работы оптимизатора](assets/demo.gif)

## 📌 Архитектура проекта

<p align="center">
  <img src="https://i.ibb.co/G3scrSyR/IMAGE-2025-04-29-19-38-11.jpg" width="30%">
</p>

## Пример результата работы пайплайна на случайном наборе параметров:

**Example Pipeline Run Output Tree (Concise):**

*   [`run_20250429_174251_balanced_portfolio_0429_1715/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715)
    *   [`pipeline_summary.md`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/pipeline_summary.md) - Общий итог пайплайна
    *   [`bonds/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/bonds/) - Анализ и портфель облигаций
        *   [`portfolio/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/bonds/portfolio/) - Результаты (статистика, веса, графики)
        *   [`reports/bonds_kbd_report_...html`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/bonds/reports/bonds_kbd_report_20250429_174251.html) - HTML отчет
    *   [`portfolio/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/portfolio/) - Оптимизация портфеля акций (лонг)
        *   [`black_litterman/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/portfolio/black_litterman/) - Результаты Black-Litterman
        *   [`markowitz/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/portfolio/markowitz/) - Результаты Марковица
    *   [`signals/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/signals/) - Генерация сигналов (для шорта)
        *   [`ticker_visualizations/AFKS/signals.png`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/signals/ticker_visualizations/AFKS/signals.png) - График сигналов (Пример: AFKS)
        *   [`ticker_visualizations/AFKS/scores.png`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/signals/ticker_visualizations/AFKS/scores.png) - График скоринга (Пример: AFKS)
    *   [`shorts_portfolio/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/shorts_portfolio/) - Шорт-стратегия и бэктест
        *   [`honest_backtest/honest_backtest_report.md`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/shorts_portfolio/honest_backtest/honest_backtest_report.md) - Отчет по бэктесту OOS
        *   [`production_portfolio/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/shorts_portfolio/production_portfolio/) - Финальный шорт-портфель
    *   [`final_portfolio/`](https://github.com/aeshef/kursach/tree/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/final_portfolio/) - Итоговый скомбинированный портфель
        *   [`README.md`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/final_portfolio/README.md) - Описание итогового портфеля
        *   [`combined_portfolio_pie.png`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/final_portfolio/combined_portfolio_pie.png) - Структура итогового портфеля (Pie)
        *   [`combined_portfolio_bars.png`](https://github.com/aeshef/kursach/blob/main/data/pipeline_runs/run_20250429_174251_balanced_portfolio_0429_1715/final_portfolio/combined_portfolio_bars.png) - Структура итогового портфеля (Bars)


## 🚀 Инструкция по запуску

## Общие требования

- Python 3.8+ (настоятельно рекомендуется 3.9.22)
- (опционально) Jupyter Notebook или Visual Studio Code

## 1. Клонируй репозиторий

```bash
git clone https://github.com/aeshef/bRECs.git
cd bRECs
```

## 2. Создай и активируй виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(на Windows: .\.venv\Scripts\activate)*

---

## 3. Установи зависимости и проект в editable-режиме

```bash
pip install --upgrade pip
pip install -e .
```
Если используешь Jupyter Notebook — также:

```bash
pip install jupyter ipykernel
```

---

## 4. Добавь Jupyter kernel* (чтобы запускать ноутбуки в своем venv)

```bash
python -m ipykernel install --user --name kursach-env --display-name "bRECs env"
```

Теперь в Jupyter или в VS Code выбирай ядро "Kursach Env".

---

## 5. Как теперь правильно импортировать свой код

После выполнения команд выше, в любом ноутбуке или скрипте из любой папки проекта можно писать:

```py
from pys.portfolio_optimization.executor import run_pipeline
```

---

## ⚙️ Требования
- Python 3.8+ (настоятельно рекомендуется 3.9.22)
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- jupyter  
- ipykernel  
- sqlalchemy  
- psycopg2  
- python-telegram-bot  
- yfinance  
- requests  
- beautifulsoup4  
- pyportfolioopt  
- и другие библиотеки (см. `setup.py` и `requirements.txt`)

## 📬 Контакты
- **Почта:** liza.bolotnikova@gmail.com aeshevchenko1704@gmail.com
- **Telegram:** @liza_bolotnikova @plxlrd
