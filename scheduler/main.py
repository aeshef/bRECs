# /Твой_Проект_Kursach/scheduler/main.py
import logging
import os
import sys
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor as APSchedulerThreadPoolExecutor
from dotenv import load_dotenv
from telegram import Bot, TelegramError

# --- Настройка Путей ---
try:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
    PYS_PATH = PROJECT_ROOT / 'pys'
    if PYS_PATH.exists() and str(PYS_PATH) not in sys.path: sys.path.insert(0, str(PYS_PATH))
    from pys.utils.path_helper import get_project_root, get_logs_path
    LOGS_PATH = get_logs_path()
except ImportError:
    print("Fatal Error: Could not import path_helper in scheduler.")
    sys.exit(1)

# --- Загрузка .env ---
env_path = PROJECT_ROOT / '.env'
if env_path.exists(): load_dotenv(dotenv_path=env_path)
else: print(f"Warning: .env file not found at {env_path}")

ENV_TYPE = os.getenv('ENV_TYPE', 'local')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
try:
    ADMIN_IDS = json.loads(os.getenv('TELEGRAM_ADMIN_IDS', '[]'))
    if not isinstance(ADMIN_IDS, list): ADMIN_IDS = []
except Exception: ADMIN_IDS = []
SCHEDULER_MAX_WORKERS = int(os.getenv('SCHEDULER_MAX_WORKERS', '2')) # Кол-во потоков для юзеров

# --- Настройка Логирования ---
log_file_path = LOGS_PATH / 'scheduler.log'
logging.basicConfig(
    level=logging.INFO, # Меньше логов для планировщика
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# --- Импорты из Проекта ---
try:
    from db.models import SessionLocal, User, engine as db_engine
    from db import crud
    from core.pipeline_runner import run_global_data_update, run_user_portfolio_update
    from tg_bot.notification_sender import send_result_notification
except ImportError as e:
    logger.critical(f"Fatal Error: Could not import project modules. Error: {e}")
    sys.exit(1)

# --- Уведомление Админа ---
def notify_admin(message: str):
    if not BOT_TOKEN or not ADMIN_IDS: return
    try:
        bot = Bot(token=BOT_TOKEN)
        full_message = f"🔔 *Scheduler Alert* ({ENV_TYPE})\n\n{message}"
        for admin_id in ADMIN_IDS:
             try:
                  bot.send_message(chat_id=admin_id, text=full_message, parse_mode='Markdown')
             except TelegramError as te:
                  logger.warning(f"Failed to send alert to admin {admin_id}: {te}")
    except Exception as e:
        logger.error(f"Failed to initialize bot for admin notification: {e}")


# --- Функции Задач Планировщика ---
# Пул для параллельного запуска обновлений портфелей пользователей
# Используем имя потоков для логирования
portfolio_update_executor = ThreadPoolExecutor(max_workers=SCHEDULER_MAX_WORKERS,
                                               thread_name_prefix='PortfolioUpdateWorker')

def trigger_global_data_update():
    """Задача для запуска глобального обновления данных."""
    logger.info("SCHEDULER JOB STARTED: Running global data update...")
    start_time = time.monotonic()
    try:
        success = run_global_data_update()
        duration = time.monotonic() - start_time
        if success:
            logger.info(f"SCHEDULER JOB FINISHED: Global data update successful. Duration: {duration:.2f}s")
        else:
            logger.error("SCHEDULER JOB FAILED: Global data update returned False.")
            notify_admin("❌ Global data update job failed. Check scheduler logs.")
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.exception(f"SCHEDULER JOB FAILED: Exception during global data update job after {duration:.2f}s.")
        notify_admin(f"💥 Exception in global data update job: {e}")

def trigger_single_user_update_task(user_id: int, telegram_id: int):
     """
     Выполняет обновление для одного юзера и отправляет уведомление.
     Эта функция будет запущена в отдельном потоке из ThreadPoolExecutor.
     """
     logger.info(f"PortfolioUpdateWorker started for user_id={user_id}")
     start_time = time.monotonic()
     bot_instance = None

     try:
         # --- Запуск Ядра ---
         final_weights, final_metrics, significant_changes = run_user_portfolio_update(user_id, is_initial=False)
         duration = time.monotonic() - start_time

         # --- Отправка уведомления (только если нужно) ---
         if final_weights is not None and significant_changes:
             logger.info(f"Portfolio update success for user {user_id} with changes. Duration: {duration:.2f}s. Notifying...")
             if BOT_TOKEN:
                 try:
                     bot_instance = Bot(token=BOT_TOKEN)
                     send_result_notification(
                         bot=bot_instance, chat_id=telegram_id, success=True,
                         weights=final_weights, metrics=final_metrics,
                         is_initial=False, significant_changes=True
                     )
                     logger.info(f"Notification sent successfully to user {user_id}.")
                 except TelegramError as te:
                      logger.error(f"TelegramError sending notification to user {user_id}: {te}")
                 except Exception as notify_err:
                      logger.exception(f"Error sending notification to user {user_id}: {notify_err}")
             else:
                 logger.warning(f"BOT_TOKEN not found, cannot send notification to user {user_id}.")

         elif final_weights is None:
              logger.error(f"PortfolioUpdateWorker FAILED for user {user_id}. Duration: {duration:.2f}s. No notification sent.")
              # Можно уведомить админа
              # notify_admin(f" portfolio update failed for user_id: {user_id}")
         else:
             logger.info(f"Portfolio update for user {user_id} successful, no significant changes. Duration: {duration:.2f}s.")

     except Exception as e:
          duration = time.monotonic() - start_time
          logger.exception(f"PortfolioUpdateWorker FAILED with exception for user_id={user_id} after {duration:.2f}s.")
          # Уведомить админа?
          # notify_admin(f"💥 Exception in single user update task for user_id={user_id}: {e}")
     finally:
          logger.info(f"PortfolioUpdateWorker finished for user_id={user_id}")


def trigger_weekly_user_updates():
    """Задача для запуска еженедельного обновления портфелей всех активных юзеров."""
    logger.info("SCHEDULER JOB STARTED: Running weekly user portfolio updates...")
    start_time = time.monotonic()
    active_users = []
    db = SessionLocal()
    try:
        active_users = crud.get_active_users(db)
        logger.info(f"Found {len(active_users)} active users for weekly update.")
    except Exception as e:
        logger.exception("Error fetching active users from DB.")
        notify_admin(f"💥 Error fetching active users for weekly update: {e}")
        return
    finally:
        db.close()

    if not active_users:
        logger.info("No active users found. Skipping weekly updates.")
        return

    # Запускаем обновление для каждого пользователя в отдельном потоке
    futures = {}
    submitted_count = 0
    for user in active_users:
        if user.id and user.telegram_id:
             try:
                  future = portfolio_update_executor.submit(trigger_single_user_update_task, user.id, user.telegram_id)
                  futures[future] = user.id # Сохраняем ID для логирования ошибок
                  submitted_count += 1
             except Exception as submit_err:
                  logger.error(f"Failed to submit update task for user_id={user.id}: {submit_err}")
        else:
             logger.warning(f"Skipping user with invalid id or telegram_id: {user.id if user else 'Unknown'}")

    logger.info(f"Submitted {submitted_count} user update tasks to executor.")

    # Ожидание завершения и анализ ошибок
    errors_count = 0
    completed_count = 0
    for future in as_completed(futures):
         user_id_err = futures[future]
         try:
             future.result()
             completed_count += 1
         except Exception as exc:
             logger.error(f"Task for user_id={user_id_err} generated an exception: {exc}", exc_info=False) # Не дублируем traceback
             errors_count += 1

    duration = time.monotonic() - start_time
    logger.info(f"SCHEDULER JOB FINISHED: Weekly user updates processed. Completed: {completed_count}, Errors: {errors_count}. Total Duration: {duration:.2f}s")
    if errors_count > 0:
         notify_admin(f"⚠️ Weekly portfolio update finished with {errors_count} errors out of {submitted_count} submitted tasks.")


# --- Настройка и Запуск Планировщика ---
if __name__ == "__main__":
    logger.info(f"--- Starting Scheduler --- mode: {ENV_TYPE} ---")

    # Хранилище задач (чтобы не добавлять их заново при перезапуске)
    # Используем SQLAlchemyJobStore, указывая URL из конфига БД
    try:
        from db.models import DATABASE_URL as DB_URL_FOR_STORE # Импортируем URL из моделей
        if 'sqlite' in DB_URL_FOR_STORE:
             # Для SQLite путь должен быть абсолютным
             db_path = DB_URL_FOR_STORE.split('///')[1]
             if not os.path.isabs(db_path):
                   db_path = str(PROJECT_ROOT / db_path)
             jobstore_url = f'sqlite:///{db_path}'
        else:
             jobstore_url = DB_URL_FOR_STORE
        logger.info(f"Using database job store: {jobstore_url.split('@')[1] if '@' in jobstore_url else jobstore_url}")
        jobstores = {
            'default': SQLAlchemyJobStore(url=jobstore_url, tablename='apscheduler_jobs')
        }
    except Exception as store_err:
         logger.exception("Failed to configure SQLAlchemyJobStore. Using default MemoryJobStore.")
         jobstores = {'default': None}

    executors = {
         'default': APSchedulerThreadPoolExecutor(5)
    }
    job_defaults = {
        'coalesce': True, # Не запускать пропущенные задачи несколько раз подряд
        'max_instances': 1, # Гарантировать, что только один экземпляр задачи выполняется одновременно
        'misfire_grace_time': 3600 # 1 час на запуск пропущенной задачи
    }

    try:
         import pytz
         timezone = pytz.timezone('Europe/Moscow')
    except ImportError:
         logger.warning("pytz not installed. Using system default timezone.")
         timezone = None

    scheduler = BlockingScheduler(
         jobstores=jobstores,
         executors=executors,
         job_defaults=job_defaults,
         timezone=timezone
    )

    # --- Добавление Задач ---
    try:
        # 1. Глобальное обновление данных (ежедневно в 03:05 по МСК)
        scheduler.add_job(
            trigger_global_data_update,
            trigger='cron',
            hour=3,
            minute=5,
            id='global_data_update_daily',
            name='Daily Global Data Update',
            replace_existing=True,
        )

        # 2. Еженедельное обновление портфелей (каждый понедельник в 05:05 по МСК)
        scheduler.add_job(
             trigger_weekly_user_updates,
             trigger='cron',
             day_of_week='mon',
             hour=5,
             minute=5,
             id='weekly_user_updates_all',
             name='Weekly User Portfolio Updates',
             replace_existing=True,
             misfire_grace_time=3600 * 3 # Даем больше времени (3 часа) на выполнение этой задачи
        )

        logger.info("--- Scheduler Jobs ---")
        scheduler.print_jobs()
        logger.info("----------------------")

        logger.info("Starting scheduler event loop...")
        # Уведомление админа о старте (опционально)
        notify_admin(f"✅ Scheduler started successfully in {ENV_TYPE} mode.")

        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutdown requested.")
    except Exception as e:
        logger.critical("Fatal error starting scheduler.", exc_info=True)
        notify_admin(f"💥 FATAL ERROR starting scheduler: {e}")
    finally:
        if scheduler.running:
             logger.info("Shutting down scheduler...")
             scheduler.shutdown()
        logger.info("Shutting down portfolio update executor...")
        portfolio_update_executor.shutdown(wait=True)
        logger.info("Scheduler and executors shut down gracefully.")

