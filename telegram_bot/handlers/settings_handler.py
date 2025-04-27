import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CallbackQueryHandler, MessageHandler, Filters

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
import sys
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from database.models import SessionLocal
from database import crud

# Состояния для ConversationHandler
SETTINGS = 5
START_ROUTES = 0

def show_settings(update: Update, context: CallbackContext):
    """
    Показывает настройки пользователя
    """
    query = update.callback_query
    query.answer()
    
    user = update.effective_user
    db = SessionLocal()
    
    try:
        # Получаем данные пользователя
        db_user = crud.get_user_by_telegram_id(db, user.id)
        
        if not db_user:
            query.edit_message_text(
                "Ошибка: пользователь не найден в базе данных. Пожалуйста, перезапустите бота командой /start."
            )
            return START_ROUTES
        
        # Получаем предпочтения пользователя
        preferences = crud.get_user_preferences(db, db_user.id)
        
        # Получаем текущий риск-профиль
        risk_profile = db_user.risk_profile or "не определен"
        
        # Формируем сообщение с настройками
        message = "⚙️ *Настройки профиля*\n\n"
        
        message += f"*Риск-профиль:* {get_profile_name_russian(risk_profile)}\n\n"
        
        if preferences:
            message += "*Предпочтения портфеля:*\n"
            message += f"• Максимум акций: {preferences.max_stocks}\n"
            message += f"• Максимум облигаций: {preferences.max_bonds}\n"
            
            if preferences.excluded_sectors and len(preferences.excluded_sectors) > 0:
                message += f"• Исключенные секторы: {', '.join(preferences.excluded_sectors)}\n"
            
            if preferences.preferred_sectors and len(preferences.preferred_sectors) > 0:
                message += f"• Предпочтительные секторы: {', '.join(preferences.preferred_sectors)}\n"
        
        # Создаем кнопки
        keyboard = [
            [InlineKeyboardButton("Изменить риск-профиль", callback_data="change_risk_profile")],
            [InlineKeyboardButton("Изменить предпочтения", callback_data="change_preferences")],
            [InlineKeyboardButton("Деактивировать аккаунт", callback_data="deactivate_account")],
            [InlineKeyboardButton("Назад", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        return SETTINGS
    
    except Exception as e:
        logger.error(f"Ошибка при отображении настроек: {e}")
        query.edit_message_text(
            "Произошла ошибка при загрузке настроек. Пожалуйста, попробуйте позже."
        )
        return START_ROUTES
    
    finally:
        db.close()

def get_profile_name_russian(profile):
    """
    Возвращает название профиля на русском языке
    """
    profiles = {
        "conservative": "Консервативный",
        "moderate": "Умеренный",
        "aggressive": "Агрессивный"
    }
    return profiles.get(profile, profile)

def handle_settings_action(update: Update, context: CallbackContext):
    """
    Обрабатывает действия в разделе настроек
    """
    query = update.callback_query
    query.answer()
    
    if query.data == "change_risk_profile":
        # Показываем выбор нового риск-профиля
        keyboard = [
            [InlineKeyboardButton("Консервативный", callback_data="set_risk_conservative")],
            [InlineKeyboardButton("Умеренный", callback_data="set_risk_moderate")],
            [InlineKeyboardButton("Агрессивный", callback_data="set_risk_aggressive")],
            [InlineKeyboardButton("Назад", callback_data="settings")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            "*Выберите новый риск-профиль:*\n\n"
            "🔵 *Консервативный* - минимальный риск, стабильный доход\n"
            "🟡 *Умеренный* - сбалансированный подход к риску и доходности\n"
            "🔴 *Агрессивный* - высокий риск ради максимальной доходности\n\n"
            "⚠️ Изменение риск-профиля приведет к пересчету рекомендаций для вашего портфеля.",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        return SETTINGS
    
    elif query.data.startswith("set_risk_"):
        # Устанавливаем новый риск-профиль
        risk_profile = query.data.replace("set_risk_", "")
        
        user = update.effective_user
        db = SessionLocal()
        
        try:
            # Обновляем риск-профиль пользователя
            crud.update_user_risk_profile(db, user.id, risk_profile)
            
            query.edit_message_text(
                f"✅ Ваш риск-профиль успешно изменен на *{get_profile_name_russian(risk_profile)}*.\n\n"
                "Рекомендации по портфелю будут обновлены при следующем расчете.",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Вернуться к настройкам", callback_data="settings")],
                    [InlineKeyboardButton("Главное меню", callback_data="back_to_main")]
                ])
            )
        
        except Exception as e:
            logger.error(f"Ошибка при изменении риск-профиля: {e}")
            query.edit_message_text(
                "Произошла ошибка при изменении риск-профиля. Пожалуйста, попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Вернуться к настройкам", callback_data="settings")]
                ])
            )
        
        finally:
            db.close()
        
        return SETTINGS
    
    elif query.data == "change_preferences":
        # Показываем настройки предпочтений
        keyboard = [
            [InlineKeyboardButton("Максимум акций", callback_data="set_max_stocks")],
            [InlineKeyboardButton("Максимум облигаций", callback_data="set_max_bonds")],
            [InlineKeyboardButton("Исключить секторы", callback_data="exclude_sectors")],
            [InlineKeyboardButton("Предпочтительные секторы", callback_data="prefer_sectors")],
            [InlineKeyboardButton("Назад", callback_data="settings")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            "*Настройки предпочтений портфеля*\n\n"
            "Выберите параметр, который хотите изменить:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        return SETTINGS
    
    elif query.data == "deactivate_account":
        # Показываем подтверждение деактивации аккаунта
        keyboard = [
            [InlineKeyboardButton("Да, деактивировать", callback_data="confirm_deactivate")],
            [InlineKeyboardButton("Нет, отмена", callback_data="settings")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            "⚠️ *Вы уверены, что хотите деактивировать аккаунт?*\n\n"
            "При деактивации:\n"
            "• Вы перестанете получать уведомления и рекомендации\n"
            "• Ваши настройки и портфель будут сохранены\n"
            "• Вы сможете восстановить аккаунт позже, написав /start\n\n"
            "Хотите продолжить?",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        return SETTINGS
    
    elif query.data == "confirm_deactivate":
        # Деактивируем аккаунт пользователя
        user = update.effective_user
        db = SessionLocal()
        
        try:
            # Получаем данные пользователя
            db_user = crud.get_user_by_telegram_id(db, user.id)
            
            if db_user:
                # Деактивируем пользователя
                db_user.is_active = False
                db.commit()
                
                query.edit_message_text(
                    "✅ Ваш аккаунт успешно деактивирован.\n\n"
                    "Вы больше не будете получать уведомления и рекомендации.\n"
                    "Чтобы восстановить аккаунт, отправьте команду /start.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Вернуться в начало", callback_data="back_to_main")]
                    ])
                )
            else:
                query.edit_message_text(
                    "Ошибка: пользователь не найден в базе данных.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Вернуться в начало", callback_data="back_to_main")]
                    ])
                )
        
        except Exception as e:
            logger.error(f"Ошибка при деактивации аккаунта: {e}")
            query.edit_message_text(
                "Произошла ошибка при деактивации аккаунта. Пожалуйста, попробуйте позже.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Вернуться к настройкам", callback_data="settings")]
                ])
            )
        
        finally:
            db.close()
        
        return START_ROUTES
    
    elif query.data == "back_to_main":
        # Возврат в главное меню
        keyboard = [
            [InlineKeyboardButton("📊 Мой портфель", callback_data="portfolio")],
            [InlineKeyboardButton("🔄 Ребалансировка", callback_data="rebalance")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            "Выберите действие из главного меню:",
            reply_markup=reply_markup
        )
        
        return START_ROUTES
    
    elif query.data == "settings":
        # Возврат к настройкам
        return show_settings(update, context)
    
    return SETTINGS

def get_handlers():
    """
    Возвращает список обработчиков для состояния SETTINGS
    """
    return [
        CallbackQueryHandler(handle_settings_action)
    ]
