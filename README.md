# Инструкция по запуску и работе с проектом

## Общие требования

- Python 3.8+ (рекомендуется 3.10+)
- git
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

## 6. Как добавлять файлы в репозиторий (работа с .gitignore)

В проекте должен быть файл .gitignore в корне со следующим содержимым:

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.idea/
```

- Все временные/локальные файлы и папки из этого списка не будут попадать в репозиторий.
- В git должны попадать только код, ноутбуки, документация и инфраструктура.

---

## 7. Как коммитить и пушить (через GitHub Desktop)

1. Открой репозиторий в GitHub Desktop.
2. Убедись, что нет в изменениях папок/файлов из .venv, __pycache__, .ipynb_checkpoints, .idea, *.pyc.
   - Если появились, клик ПКМ — "Discard changes".
3. Введи осмысленное описание коммита, нажми "Commit to main" (или ветку).
4. Нажми "Push origin" для отправки изменений.

Если случайно добавил лишние файлы, удали их из индекса (см. выше) и перекоммить.

---

## 8. Как не хранить секреты в публичном git

- Все токены, секреты и приватную информацию (например, private_info.py) не клади в git.
- Вместо этого клади файл-пример: private_info.py.example (пусть коллега копирует и правит его локально).

---

## 9. Частые ошибки

- Если импорт не работает — проверь, что активен venv и что в нем прописан твой пакет (pip list)
- Перезапусти Jupyter kernel после установки зависимостей.
- Не запускать ноутбуки/скрипты из-под системного интерпретатора Python: только из venv!
