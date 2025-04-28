import os
import sys
import socket
import logging
from typing import Dict, Any, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импортируем определение путей
try:
    from pys.utils.path_helper import get_project_root, is_server
except ImportError:
    # Если не получается импортировать, определяем функции здесь
    def is_server():
        hostname = socket.gethostname()
        return hostname == '4674313-se07272'  # Имя сервера
    
    def get_project_root():
        if is_server():
            return '/opt/portfolio-advisor'
        else:
            # Определяем относительно текущего файла
            current_file = os.path.abspath(__file__)
            if 'pys' in current_file:
                parts = current_file.split('pys')
                return os.path.join(parts[0], 'pys')
            else:
                return os.path.dirname(os.path.dirname(current_file))

def load_secrets() -> Dict[str, Any]:
    """Загружает секретные данные из файла конфигурации"""
    secrets = {}
    
    # Определяем путь к файлу с секретами
    project_root = get_project_root()
    
    if is_server():
        # На сервере используем абсолютный путь
        secret_file_path = f"{project_root}/config/secrets/secret_config.py"
    else:
        # Локально ищем в pys/config/secrets
        secret_file_path = f"{project_root}/config/secrets/secret_config.py"
        if not os.path.exists(secret_file_path):
            # Пробуем альтернативный путь
            secret_file_path = f"{project_root}/pys/config/secrets/secret_config.py"
    
    # Проверяем существование файла
    if not os.path.exists(secret_file_path):
        logger.warning(f"Файл с секретами не найден: {secret_file_path}")
        logger.warning("Создайте файл на основе шаблона secret_config_template.py")
        # Пытаемся получить данные из старого файла private_info.py
        try:
            if is_server():
                # На сервере мы еще не создали этот файл
                return {}
            else:
                # Локально пытаемся импортировать из private_info
                sys.path.insert(0, project_root)
                from pys.data_collection.private_info import (
                    token as TINKOFF_TOKEN,
                    YOUR_API_ID, YOUR_API_HASH,
                    TOKEN as TELEGRAM_BOT_TOKEN,
                    ADMIN_IDS, DB_PASSWORD, BASE_PATH
                )
                
                return {
                    "TINKOFF_TOKEN": TINKOFF_TOKEN,
                    "TELEGRAM_API_ID": YOUR_API_ID,
                    "TELEGRAM_API_HASH": YOUR_API_HASH,
                    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
                    "ADMIN_IDS": ADMIN_IDS if isinstance(ADMIN_IDS, list) else [ADMIN_IDS],
                    "DB_PASSWORD": DB_PASSWORD,
                    "BASE_PATH": BASE_PATH
                }
        except ImportError:
            logger.error("Не удалось импортировать данные из private_info.py")
            return {}
    
    # Загружаем модуль с секретами
    sys.path.insert(0, os.path.dirname(os.path.dirname(secret_file_path)))
    try:
        # Пытаемся импортировать из файла secret_config.py
        if 'pys' in secret_file_path:
            module_path = 'pys.config.secrets.secret_config'
        else:
            module_path = 'config.secrets.secret_config'
        
        # Динамический импорт
        from importlib import import_module
        secret_module = import_module(module_path)
        
        # Получаем значения из модуля
        secrets = {
            "TINKOFF_TOKEN": getattr(secret_module, "TINKOFF_TOKEN", ""),
            "TELEGRAM_API_ID": getattr(secret_module, "TELEGRAM_API_ID", 0),
            "TELEGRAM_API_HASH": getattr(secret_module, "TELEGRAM_API_HASH", ""),
            "TELEGRAM_BOT_TOKEN": getattr(secret_module, "TELEGRAM_BOT_TOKEN", ""),
            "ADMIN_IDS": getattr(secret_module, "ADMIN_IDS", []),
            "DB_PASSWORD": getattr(secret_module, "DB_PASSWORD", "")
        }
        
        # Добавляем пути к данным
        if is_server():
            secrets["BASE_PATH"] = "/opt/portfolio-advisor/data"
        else:
            # Пытаемся получить из private_info.py или используем корень проекта
            try:
                from pys.data_collection.private_info import BASE_PATH
                secrets["BASE_PATH"] = BASE_PATH
            except ImportError:
                secrets["BASE_PATH"] = project_root
            
        logger.info("Секретные данные успешно загружены")
        return secrets
        
    except ImportError as e:
        logger.error(f"Ошибка при импорте секретных данных: {e}")
        return {}

# Глобальный словарь с секретами для использования в приложении
SECRETS = load_secrets()

def get_secret(key: str, default: Optional[Any] = None) -> Any:
    """Получает значение секрета по ключу"""
    return SECRETS.get(key, default)
