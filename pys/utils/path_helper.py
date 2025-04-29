# import os
# import sys
# import socket

# def add_project_root_to_path():
#     """Add project root to Python path to enable imports across modules"""
#     # Получаем путь к корню проекта (на 3 уровня выше, чем текущий файл)
#     # utils/path_helper.py -> utils -> pys -> project_root
#     project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    
#     # Проверяем, есть ли уже корень проекта в пути
#     if project_root not in sys.path:
#         sys.path.insert(0, project_root)
    
#     # Также добавляем путь к директории pys, чтобы можно было импортировать оттуда
#     pys_path = os.path.join(project_root, 'pys')
#     if pys_path not in sys.path:
#         sys.path.insert(0, pys_path)
        
#     return project_root

# import os
# import sys
# import socket

# def get_project_root():
#     """Определяет корневой каталог проекта в зависимости от окружения"""
#     hostname = socket.gethostname()
#     server_hostname = '4674313-se07272'  # Имя вашего сервера из команды hostname
    
#     if hostname == server_hostname:
#         return '/opt/portfolio-advisor'
#     else:
#         # Для локальной разработки
#         # Определяем путь к корню проекта (директория, содержащая pys)
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         # Поднимаемся на два уровня вверх: utils -> pys -> корень проекта
#         if 'pys' in current_dir:
#             root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
#             return root_dir
#         else:
#             # Запасной вариант
#             return os.path.dirname(os.path.dirname(current_dir))

# def setup_python_path():
#     """Добавляет корневой каталог проекта в sys.path"""
#     root = get_project_root()
#     if root not in sys.path:
#         sys.path.insert(0, root)
    
#     # Также добавляем директорию pys для импортов
#     pys_path = os.path.join(root, 'pys')
#     if os.path.exists(pys_path) and pys_path not in sys.path:
#         sys.path.insert(0, pys_path)
    
#     return root

# def is_server():
#     """Проверяет, выполняется ли код на сервере"""
#     hostname = socket.gethostname()
#     server_hostname = '4674313-se07272'
#     return hostname == server_hostname

# def get_base_path():
#     """Возвращает BASE_PATH в зависимости от окружения"""
#     if is_server():
#         return '/opt/portfolio-advisor/data'
#     else:
#         # Для локальной разработки используем оригинальный путь
#         try:
#             # Пытаемся импортировать напрямую
#             sys.path.insert(0, get_project_root())
#             from pys.data_collection.private_info import BASE_PATH
#             return BASE_PATH
#         except ImportError:
#             # Запасной вариант - возвращаем корень проекта
#             return get_project_root()

from pathlib import Path
import os
import sys

# --- Определение Корня Проекта ---
# Идем на 3 уровня вверх от текущего файла: /pys/utils/path_helper.py -> /pys/utils -> /pys -> /
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    # Проверка: ищем характерный файл/папку в корне, например, '.env' или 'pys'
    if not (PROJECT_ROOT / '.env').is_file() and not (PROJECT_ROOT / 'pys').is_dir():
         # Если не нашли, возможно, структура другая? Попробуем другой метод.
         # Эта часть может потребовать адаптации, если твоя структура отличается.
         print("Warning: PROJECT_ROOT auto-detection might be inaccurate. Trying parent.")
         alt_root = Path(os.getcwd()) # Запасной вариант - текущая рабочая директория? Не очень надежно.
         if (alt_root / 'pys').is_dir():
              PROJECT_ROOT = alt_root
         else:
              # Если ничего не помогает, используем расчетный путь, но выводим предупреждение
              print(f"Warning: Assuming project root is {PROJECT_ROOT}, but validation failed.")
except NameError:
     # Если __file__ не определен (например, в интерактивной сессии)
     print("Warning: __file__ not defined. Using current working directory as PROJECT_ROOT.")
     PROJECT_ROOT = Path(os.getcwd()).resolve()

# --- Функции для получения путей ---

def get_project_root() -> Path:
    """Возвращает объект Path корневой директории проекта."""
    return PROJECT_ROOT

def get_pys_path() -> Path:
    """Возвращает путь к директории 'pys'."""
    path = PROJECT_ROOT / 'pys'
    path.mkdir(parents=True, exist_ok=True) # Создаем, если нет
    return path

def get_data_path() -> Path:
    """Возвращает путь к основной директории 'data' в корне проекта."""
    path = PROJECT_ROOT / 'data'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logs_path() -> Path:
     """Возвращает путь к основной директории 'logs' в корне проекта."""
     path = PROJECT_ROOT / 'logs'
     path.mkdir(parents=True, exist_ok=True)
     return path

def get_scripts_path() -> Path:
    """Возвращает путь к директории 'scripts'."""
    path = PROJECT_ROOT / 'scripts'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_db_path() -> Path:
    """Возвращает путь к директории 'db'."""
    path = PROJECT_ROOT / 'db'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_core_path() -> Path:
    """Возвращает путь к директории 'core'."""
    path = PROJECT_ROOT / 'core'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_tg_bot_path() -> Path:
    """Возвращает путь к директории 'tg_bot'."""
    path = PROJECT_ROOT / 'tg_bot'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_scheduler_path() -> Path:
    """Возвращает путь к директории 'scheduler'."""
    path = PROJECT_ROOT / 'scheduler'
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_venv_path() -> Path:
     """Возвращает путь к директории 'venv' (если она в корне)."""
     return PROJECT_ROOT / 'venv'


# --- Настройка sys.path ---
def setup_python_path():
    """Добавляет корень проекта и папку pys в sys.path для импортов."""
    project_root_str = str(PROJECT_ROOT)
    pys_path_str = str(get_pys_path())

    # Добавляем пути в начало sys.path, если их там еще нет
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        # print(f"Added to sys.path: {project_root_str}") # Для отладки
    if pys_path_str != project_root_str and pys_path_str not in sys.path:
        sys.path.insert(0, pys_path_str)
        # print(f"Added to sys.path: {pys_path_str}") # Для отладки

# Вызываем настройку пути при импорте модуля.
# Это упрощает запуск скриптов из разных мест.
setup_python_path()

# --- Старая логика (для справки, но не используется) ---
# Функции is_server, get_project_root (старая), get_base_path (старая)
# основанные на hostname, удалены как ненадежные.
# Импорт BASE_PATH из private_info удален как плохая практика.
