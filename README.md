# 📊 Название проекта

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

<img src="https://i.ibb.co/G3scrSyR/IMAGE-2025-04-29-19-38-11.jpg" width="50%">

## 🚀 Инструкция по запуску

## Общие требования

- Python 3.8+ (настоятельбно рекомендуется 3.9.22)
- (опционально) Jupyter Notebook или Visual Studio Code

## 1. Клонируй репозиторий

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd название_папки_проекта
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
python -m ipykernel install --user --name kursach-env --display-name "Kursach Env"
```

Теперь в Jupyter или в VS Code выбирай ядро "Kursach Env".

---

## 5. Как теперь правильно импортировать свой код

После выполнения команд выше, в любом ноутбуке или скрипте из любой папки проекта можно писать:

```py
from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH
from pys.data_collection.bonds_processor import run_pipeline_bonds_processor
```

и т.п.
НЕ используйте sys.path.insert, os.chdir и т.п. костыли — они больше не нужны и только мешают!

---

## ⚙️ Требования
- Python 3.8+(настоятельбно рекомендуется 3.9.22)
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
- и другие библиотеки (см. `setup.py`)

## 📬 Контакты
- **Почта:** liza.bolotnikova@gmail.com aeshevchenko1704@gmail.com
- **Telegram:** @liza_bolotnikova @plxlrd
