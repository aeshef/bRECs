import inspect
import collections
import os
import sys
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_patch():
    if not hasattr(inspect, 'getargspec'):
        ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
        inspect.ArgSpec = ArgSpec
        
        def getargspec(func):
            sig = inspect.signature(func)
            parameters = sig.parameters
            args = []
            varargs = None
            keywords = None
            defaults = []
            
            for name, param in parameters.items():
                if param.kind == inspect.Parameter.POSITIONAL_ONLY or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(name)
                    if param.default != inspect.Parameter.empty:
                        defaults.append(param.default)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    varargs = name
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    keywords = name
                    
            if not defaults:
                defaults = None
                
            return ArgSpec(args=args, varargs=varargs, keywords=keywords, defaults=defaults)
        
        inspect.getargspec = getargspec
        print("Monkey patch applied: inspect.getargspec added")

def apply_patches():
    """
    Применяет все необходимые патчи для совместимости между
    локальной разработкой и сервером
    """
    # Определяем, где выполняется код
    hostname = socket.gethostname()
    is_server = hostname == '4674313-se07272'  # Используйте имя вашего сервера
    
    # Определяем корневые пути
    if is_server:
        base_path = '/opt/portfolio-advisor'
        data_path = '/opt/portfolio-advisor/data'
    else:
        # Ищем оригинальный BASE_PATH
        try:
            # Динамический поиск private_info
            current_file = os.path.abspath(__file__)
            if 'pys' in current_file:
                parts = current_file.split('pys')
                repo_root = parts[0].rstrip('/')
                pys_path = os.path.join(repo_root, 'pys')
                
                # Добавляем пути для импорта
                if pys_path not in sys.path:
                    sys.path.insert(0, pys_path)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                
                # Импортируем private_info
                from data_collection.private_info import BASE_PATH
                base_path = BASE_PATH
                data_path = BASE_PATH
            else:
                # Запасной вариант
                base_path = os.path.dirname(os.path.dirname(current_file))
                data_path = base_path
        except ImportError:
            # Запасной вариант, если private_info не найден
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = base_path
    
    # Добавляем пути для импорта
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    
    if 'pys' in base_path:
        # Если мы в репозитории, также добавляем pys
        pys_path = os.path.join(base_path, 'pys')
        if os.path.exists(pys_path) and pys_path not in sys.path:
            sys.path.insert(0, pys_path)
    
    logger.info(f"Путь к проекту настроен: {base_path}")
    
    # Возвращаем настроенные пути
    return base_path, data_path
