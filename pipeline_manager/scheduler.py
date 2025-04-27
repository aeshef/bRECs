from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from datetime import datetime
import logging
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/portfolio-advisor/logs/scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
sys.path.append('/opt/portfolio-advisor')

# Импортируем функции пайплайна
from pipeline_manager.weekly_pipeline import run_weekly_pipeline
from pipeline_manager.urgent_signals import check_urgent_signals

# Создаем хранилище для заданий
jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:////opt/portfolio-advisor/scheduler.sqlite')
}

# Инициализируем планировщик
scheduler = BlockingScheduler(jobstores=jobstores)

def init_scheduler():
    """Инициализация планировщика с основными задачами"""
    logger.info("Инициализация планировщика задач...")
    
    # Еженедельный запуск пайплайна (каждый понедельник в 4:00)
    scheduler.add_job(
        run_weekly_pipeline, 
        'cron', 
        day_of_week='mon', 
        hour=4, 
        minute=0,
        id='weekly_pipeline',
        replace_existing=True,
        name='Weekly Portfolio Update'
    )
    
    # Ежедневная проверка экстренных сигналов (в 10:00 утра)
    scheduler.add_job(
        check_urgent_signals, 
        'cron', 
        hour=10, 
        minute=0,
        id='urgent_signals',
        replace_existing=True,
        name='Daily Urgent Signals Check'
    )
    
    logger.info("Планировщик задач инициализирован успешно.")

if __name__ == "__main__":
    try:
        init_scheduler()
        logger.info("Запуск планировщика...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Планировщик остановлен.")
    except Exception as e:
        logger.error(f"Ошибка при запуске планировщика: {e}")
