import logging
import sys
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler, Filters, CallbackContext

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/portfolio-advisor/logs/telegram_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from config.bot_config import BOT_CONFIG
from database.models import SessionLocal
from database import crud
from telegram_bot.handlers import risk_profile_handler, portfolio_handler, settings_handler

# Состояния для ConversationHandler
(
    START_ROUTES, RISK_ASSESSMENT, RISK_PROFILE_SETUP, PORTFOLIO_VIEW, 
    PORTFOLIO_REBALANCE, SETTINGS, CONTACT_ADMIN
) = range(7)

def start(update: Update, context: CallbackContext):
    """
    Начальное приветствие и проверка, зарегистрирован ли пользователь
    """
    user = update.effective_user
    db = SessionLocal()
    
    try:
        # Проверяем, зарегистрирован ли пользователь
        db_user = crud.get_user_by_telegram_id(db, user.id)
        
        if not db_user:
            # Регистрируем нового пользователя
            db_user = crud.create_user(
                db, 
                telegram_id=user.id,
                username=user.username or "",
                first_name=user.first_name or "",
                last_name=user.last_name or ""
            )
            
            # Приветствие для нового пользователя
            update.message.reply_text(
                f"Добро пожаловать, {user.first_name}! 👋\n\n"
                "Я бот-помощник для управления вашим инвестиционным портфелем. "
                "Прежде чем начать, давайте определим ваш инвестиционный профиль."
            )
            
            # Переходим к определению риск-профиля
            return risk_profile_handler.start_risk_assessment(update, context)
        
        else:
            # Приветствие для существующего пользователя
            keyboard = [
                [InlineKeyboardButton("📊 Мой портфель", callback_data='portfolio')],
                [InlineKeyboardButton("🔄 Ребалансировка", callback_data='rebalance')],
                [InlineKeyboardButton("⚙️ Настройки", callback_data='settings')],
                [InlineKeyboardButton("ℹ️ Помощь", callback_data='help')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            update.message.reply_text(
                f"С возвращением, {user.first_name}! 👋\n\n"
                "Что бы вы хотели сделать?",
                reply_markup=reply_markup
            )
            
            return START_ROUTES
    
    except Exception as e:
        logger.error(f"Ошибка при обработке команды /start: {e}")
        update.message.reply_text(
            "Произошла ошибка при запуске бота. Пожалуйста, попробуйте еще раз позже или "
            "свяжитесь с администратором."
        )
        return ConversationHandler.END
    
    finally:
        db.close()

def button_handler(update: Update, context: CallbackContext):
    """
    Обработчик нажатий на кнопки
    """
    query = update.callback_query
    query.answer()
    
    if query.data == 'portfolio':
        return portfolio_handler.show_portfolio(update, context)
    
    elif query.data == 'rebalance':
        return portfolio_handler.show_rebalance(update, context)
    
    elif query.data == 'settings':
        return settings_handler.show_settings(update, context)
    
    elif query.data == 'help':
        query.edit_message_text(
            "📚 *Справка по командам:*\n\n"
            "/start - Начало работы и главное меню\n"
            "/portfolio - Просмотр текущего портфеля\n"
            "/rebalance - Рекомендации по ребалансировке\n"
            "/settings - Настройки профиля и предпочтений\n"
            "/help - Эта справка\n\n"
            "Если у вас возникли вопросы или проблемы, напишите /contact для связи с администратором.",
            parse_mode='Markdown'
        )
        return START_ROUTES
    
    return START_ROUTES

def help_command(update: Update, context: CallbackContext):
    """
    Команда /help - показывает справку
    """
    update.message.reply_text(
        "📚 *Справка по командам:*\n\n"
        "/start - Начало работы и главное меню\n"
        "/portfolio - Просмотр текущего портфеля\n"
        "/rebalance - Рекомендации по ребалансировке\n"
        "/settings - Настройки профиля и предпочтений\n"
        "/help - Эта справка\n\n"
        "Если у вас возникли вопросы или проблемы, напишите /contact для связи с администратором.",
        parse_mode='Markdown'
    )
    return START_ROUTES

def contact_command(update: Update, context: CallbackContext):
    """
    Команда /contact - предлагает связаться с администратором
    """
    update.message.reply_text(
        "Для связи с администратором отправьте ваше сообщение после этого текста. "
        "Оно будет переслано администратору, и он свяжется с вами при необходимости."
    )
    return CONTACT_ADMIN

def forward_to_admin(update: Update, context: CallbackContext):
    """
    Пересылает сообщение пользователя администратору
    """
    user = update.effective_user
    message = update.message.text
    
    for admin_id in BOT_CONFIG['admin_ids']:
        try:
            context.bot.send_message(
                chat_id=admin_id,
                text=f"Сообщение от пользователя:\n"
                     f"ID: {user.id}\n"
                     f"Имя: {user.first_name} {user.last_name or ''}\n"
                     f"Username: @{user.username or 'нет'}\n\n"
                     f"Сообщение: {message}"
            )
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения администратору {admin_id}: {e}")
    
    update.message.reply_text(
        "Ваше сообщение отправлено администратору! Он свяжется с вами при необходимости."
    )
    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext):
    """
    Отмена текущей операции и возврат в главное меню
    """
    update.message.reply_text("Операция отменена. Возврат в главное меню.")
    
    # Отправляем кнопки главного меню
    keyboard = [
        [InlineKeyboardButton("📊 Мой портфель", callback_data='portfolio')],
        [InlineKeyboardButton("🔄 Ребалансировка", callback_data='rebalance')],
        [InlineKeyboardButton("⚙️ Настройки", callback_data='settings')],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data='help')]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        "Что бы вы хотели сделать?",
        reply_markup=reply_markup
    )
    
    return START_ROUTES

def error_handler(update: Update, context: CallbackContext):
    """
    Обработчик ошибок
    """
    logger.error(f"Ошибка при обработке обновления {update}: {context.error}")
    
    try:
        # Отправляем сообщение об ошибке пользователю
        if update and update.effective_chat:
            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз позже."
            )
    
    except Exception as e:
        logger.error(f"Ошибка при отправке уведомления об ошибке: {e}")

def main():
    """Запуск бота"""
    updater = Updater(BOT_CONFIG['token'])
    dispatcher = updater.dispatcher
    
    # Создаем ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            START_ROUTES: [
                CallbackQueryHandler(button_handler)
            ],
            # Здесь будут добавлены другие состояния из импортированных обработчиков
            RISK_ASSESSMENT: risk_profile_handler.get_handlers(),
            RISK_PROFILE_SETUP: risk_profile_handler.get_setup_handlers(),
            PORTFOLIO_VIEW: portfolio_handler.get_view_handlers(),
            PORTFOLIO_REBALANCE: portfolio_handler.get_rebalance_handlers(),
            SETTINGS: settings_handler.get_handlers(),
            CONTACT_ADMIN: [
                MessageHandler(Filters.text & ~Filters.command, forward_to_admin)
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    # Добавляем обработчики
    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("contact", contact_command))
    
    # Обработчик ошибок
    dispatcher.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("Бот запущен")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
