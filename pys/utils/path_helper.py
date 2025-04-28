import os
import sys
import socket

def add_project_root_to_path():
    """Add project root to Python path to enable imports across modules"""
    # Получаем путь к корню проекта (на 3 уровня выше, чем текущий файл)
    # utils/path_helper.py -> utils -> pys -> project_root
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    
    # Проверяем, есть ли уже корень проекта в пути
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Также добавляем путь к директории pys, чтобы можно было импортировать оттуда
    pys_path = os.path.join(project_root, 'pys')
    if pys_path not in sys.path:
        sys.path.insert(0, pys_path)
        
    return project_root

import os
import sys
import socket

def get_project_root():
    """Определяет корневой каталог проекта в зависимости от окружения"""
    hostname = socket.gethostname()
    server_hostname = '4674313-se07272'  # Имя вашего сервера из команды hostname
    
    if hostname == server_hostname:
        return '/opt/portfolio-advisor'
    else:
        # Для локальной разработки
        # Определяем путь к корню проекта (директория, содержащая pys)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Поднимаемся на два уровня вверх: utils -> pys -> корень проекта
        if 'pys' in current_dir:
            root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
            return root_dir
        else:
            # Запасной вариант
            return os.path.dirname(os.path.dirname(current_dir))

def setup_python_path():
    """Добавляет корневой каталог проекта в sys.path"""
    root = get_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    
    # Также добавляем директорию pys для импортов
    pys_path = os.path.join(root, 'pys')
    if os.path.exists(pys_path) and pys_path not in sys.path:
        sys.path.insert(0, pys_path)
    
    return root

def is_server():
    """Проверяет, выполняется ли код на сервере"""
    hostname = socket.gethostname()
    server_hostname = '4674313-se07272'
    return hostname == server_hostname

def get_base_path():
    """Возвращает BASE_PATH в зависимости от окружения"""
    if is_server():
        return '/opt/portfolio-advisor/data'
    else:
        # Для локальной разработки используем оригинальный путь
        try:
            # Пытаемся импортировать напрямую
            sys.path.insert(0, get_project_root())
            from pys.data_collection.private_info import BASE_PATH
            return BASE_PATH
        except ImportError:
            # Запасной вариант - возвращаем корень проекта
            return get_project_root()
