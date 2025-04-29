import logging
import os
import sys
import json
from pathlib import Path
from telegram.ext import (Updater, CommandHandler, CallbackQueryHandler,
                          ConversationHandler, MessageHandler, Filters, CallbackContext, Dispatcher)
from dotenv import load_dotenv

from tg_bot.handlers import common, profile, portfolio, settings
from tg_bot.states import (START_ROUTES, RISK_ASSESSMENT, PORTFOLIO_VIEW, 
                           SETTINGS_MAIN, SETTINGS_PREFERENCES, 
                           CONTACT_ADMIN, DEACTIVATE_CONFIRM)

try:
    from pys.utils.path_helper import get_project_root, get_logs_path
    PROJECT_ROOT = get_project_root()
    LOGS_PATH = get_logs_path()
except ImportError:
    print("Fatal Error: Cannot import path_helper. Ensure project structure and sys.path are correct.")
    sys.exit(1)

env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ENV_TYPE = os.getenv('ENV_TYPE', 'local')

log_file_path = LOGS_PATH / 'telegram_bot.log'
logging.basicConfig(
    level=logging.INFO if ENV_TYPE == 'production' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured. Log file: {log_file_path}")

def main():
    if not BOT_TOKEN:
        logger.critical("!!! TELEGRAM_BOT_TOKEN not found in .env file. Bot cannot start.")
        return

    logger.info(f"Starting bot in {ENV_TYPE} mode...")
    updater = Updater(BOT_TOKEN)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', common.start)],
        states={
            START_ROUTES: common.get_start_routes_handlers(),
            CONTACT_ADMIN: common.get_contact_admin_handlers(),
            DEACTIVATE_CONFIRM: settings.get_deactivate_confirm_handlers(),
            RISK_ASSESSMENT: profile.get_risk_assessment_handlers(),
            PORTFOLIO_VIEW: portfolio.get_portfolio_view_handlers(),
            SETTINGS_MAIN: settings.get_settings_main_handlers(),
            SETTINGS_PREFERENCES: settings.get_settings_preferences_handlers(),
        },

        
        fallbacks=[
              CommandHandler('cancel', common.cancel),
              CommandHandler('start', common.start)
        ],
        # persistent=True, name="main_conversation" # Опционально для сохранения состояния между перезапусками
    )

    dispatcher.add_handler(conv_handler)

    dispatcher.add_handler(CommandHandler("help", common.help_command))
    dispatcher.add_handler(CommandHandler("contact", common.contact_command))
    dispatcher.add_handler(CommandHandler("portfolio", portfolio.show_portfolio))

    dispatcher.add_error_handler(common.error_handler)

    try:
        logger.info("Bot started polling...")
        updater.start_polling()
        logger.info("Bot is running.")
        updater.idle()
    except Exception as e:
         logger.critical(f"Fatal error during bot execution: {e}", exc_info=True)
    finally:
         logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
