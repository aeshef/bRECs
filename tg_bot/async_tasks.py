import logging
import os
import sys
from pathlib import Path
from telegram import Bot, TelegramError

try:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    PYS_PATH = PROJECT_ROOT / 'pys'
    if PYS_PATH.exists() and str(PYS_PATH) not in sys.path:
         sys.path.insert(0, str(PYS_PATH))
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists(): load_dotenv(dotenv_path=env_path)
except ImportError:
     print("Warning: Could not setup sys.path in async_tasks.")

logger = logging.getLogger(__name__)

try:
    from core.pipeline_runner import run_user_portfolio_update
    from tg_bot.notification_sender import send_result_notification
except ImportError as e:
    logger.critical(f"Fatal Error: Could not import core/db modules in async_tasks. Error: {e}")
    run_user_portfolio_update = None # Заглушка
    send_result_notification = None # Заглушка


def run_pipeline_and_notify_user(user_id_db, chat_id, is_initial):
    """
    Обертка для запуска в `run_async`. Выполняет пайплайн и уведомляет пользователя.
    Параметры:
        - user_id_db (int): ID пользователя в базе данных.
        - chat_id (int): ID чата для отправки уведомления.
        - is_initial (bool): Флаг первичной генерации.
    """
    if run_user_portfolio_update is None or send_result_notification is None:
        logger.error("Core functions (run_user_portfolio_update or send_result_notification) not imported. Async task cannot run.")
        return

    bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))

    logger.info(f"Async task started: portfolio {'initial generation' if is_initial else 'update'} for user_id {user_id_db} (chat_id: {chat_id})")

    success_flag = False
    final_weights = None
    final_metrics = None
    significant_changes = False
    report = None

    try:
        final_weights, final_metrics, significant_changes, report = run_user_portfolio_update(
             user_id=user_id_db,
             is_initial=is_initial
        )

        if final_weights is not None:
             success_flag = True
             logger.info(f"Async task pipeline execution successful for user {user_id_db}.")
        else:
             success_flag = False
             logger.error(f"Async task pipeline execution failed for user {user_id_db}.")

    except Exception as e:
        logger.exception(f"Exception during async pipeline execution for user {user_id_db}: {e}")
        success_flag = False

    try:
        send_result_notification(
            bot=bot,
            chat_id=chat_id,
            success=success_flag,
            weights=final_weights,
            metrics=final_metrics,
            is_initial=is_initial,
            significant_changes=significant_changes,
            report=report
        )
        logger.info(f"Result notification sent for async task, user {user_id_db}.")
    except TelegramError as te:
        logger.error(f"TelegramError sending result notification: {te}")
    except Exception as notify_err:
        logger.exception(f"Failed to send result notification: {notify_err}")

    logger.info(f"Task finished for user_id {user_id_db}")
