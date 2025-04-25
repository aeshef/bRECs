import json
import os
import subprocess
import sys
import datetime
from typing import Any, Dict, List, Optional, Type, Union

from pys.data_collection.private_info import BASE_PATH

def run_with_config(
    script_path: str,
    class_or_function: Optional[str] = None,
    method: Optional[str] = None,
    base_dir: str = f'{BASE_PATH}/data/meta',
    config_filename: str = 'config.json',
    **kwargs
) -> int:
    """
    Универсальная функция для запуска Python-скрипта с параметрами через конфигурационный файл.
    
    Args:
        script_path: Путь к Python-скрипту, который нужно запустить
        class_or_function: Имя класса или функции для вызова (опционально)
        method: Имя метода класса для вызова (для классов)
        base_dir: Базовая директория для сохранения конфигурационного файла
        config_filename: Имя файла конфигурации
        **kwargs: Любые параметры, которые нужно передать в скрипт
        
    Returns:
        Код возврата процесса (0 - успешное выполнение)
    """
    processed_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, datetime.date) or isinstance(v, datetime.datetime):
            processed_kwargs[k] = v.isoformat().split('T')[0]
        else:
            processed_kwargs[k] = v
    
    config = {
        "params": processed_kwargs
    }
    
    if class_or_function:
        config["target"] = class_or_function
    if method:
        config["method"] = method
    
    config_path = os.path.join(base_dir, config_filename)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Конфигурация сохранена в {config_path}")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Скрипт {script_path} не найден")
    
    command = [sys.executable, script_path, '--config', config_path]
    
    print("Запуск команды:", " ".join(command))
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        print("\nПроцесс успешно завершен")
    else:
        print(f"\nОшибка при выполнении процесса, код: {process.returncode}")
    
    return process.returncode
