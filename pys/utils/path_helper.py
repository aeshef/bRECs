import os
import sys

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
