# 📊 Название проекта

![Логотип или демонстрация работы](https://i.ibb.co/pVLH0Ky/IMAGE-2025-04-28-15-06-10.jpg)

## 📑 Оглавление
- [Описание](#описание)
- [Инструкция по запуску](#инструкция-по-запуску)
- [Пример использования](#пример-использования)
- [Документация](#документация)
- [Требования](#требования)
- [Контакты](#контакты)

## 📝 Описание
Кратко расскажи, что делает твоя программа/проект, зачем он и в чем его особенности.

## 🚀 Инструкция по запуску

## Общие требования

- Python 3.8+ (настоятельбно рекомендуется 3.9.22)
- (опционально) Jupyter Notebook или Visual Studio Code

## 1. Клонируй репозиторий

```
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd название_папки_проекта
```

## 2. Создай и активируй виртуальное окружение

```
python3 -m venv .venv
source .venv/bin/activate
```

*(на Windows: .\.venv\Scripts\activate)*

---

## 3. Установи зависимости и проект в editable-режиме

```
pip install --upgrade pip
pip install -e .
```
Если используешь Jupyter Notebook — также:

```
pip install jupyter ipykernel
```

---

## 4. Добавь Jupyter kernel* (чтобы запускать ноутбуки в своем venv)

```
python -m ipykernel install --user --name kursach-env --display-name "Kursach Env"
```

Теперь в Jupyter или в VS Code выбирай ядро "Kursach Env".

---

## 5. Как теперь правильно импортировать свой код

После выполнения команд выше, в любом ноутбуке или скрипте из любой папки проекта можно писать:

```
from pys.utils.logger import BaseLogger
from pys.data_collection.private_info import BASE_PATH
from pys.data_collection.bonds_processor import run_pipeline_bonds_processor
```

# и т.п.
**НЕ используйте sys.path.insert, os.chdir и т.п. костыли — они больше не нужны и только мешают!**

---
