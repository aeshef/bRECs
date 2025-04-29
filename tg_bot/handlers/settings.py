import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CallbackQueryHandler, ConversationHandler

from db import crud, get_db_session, UserPreferences
from tg_bot.states import (SETTINGS_MAIN, SETTINGS_PREFERENCES, START_ROUTES, 
                           RISK_ASSESSMENT, DEACTIVATE_CONFIRM)
from .profile import start_risk_assessment

logger = logging.getLogger(__name__)

def deactivate_user(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    user_tg = update.effective_user
    logger.info(f"Deactivating user {user_tg.id}")

    with get_db_session() as db:
        try:
            crud.update_user_active_status(db, user_tg.id, False)
            query.edit_message_text(
                "Ваш аккаунт деактивирован. Вы больше не будете получать уведомления.\n"
                "Чтобы вернуться, просто напишите /start."
            )
            return ConversationHandler.END 
        except Exception as e:
            logger.exception(f"Error deactivating user {user_tg.id}: {e}")
            query.edit_message_text("Произошла ошибка при деактивации. Попробуйте позже.")
            query.edit_message_reply_markup(reply_markup=get_settings_keyboard())
            return SETTINGS_MAIN

def get_deactivate_confirm_handlers():
    return [
        CallbackQueryHandler(deactivate_user, pattern='^confirm_deactivate_yes$'),
        CallbackQueryHandler(show_settings, pattern='^main_settings$')
    ]

def get_deactivate_confirm_handlers():
    return [
        CallbackQueryHandler(deactivate_user, pattern='^confirm_deactivate_yes$'),
        CallbackQueryHandler(show_settings, pattern='^main_settings$')
    ]

def get_settings_keyboard():
    """Возвращает клавиатуру настроек."""
    keyboard = [
        [InlineKeyboardButton("🎓 Изменить риск-профиль", callback_data='settings_change_profile')],
        [InlineKeyboardButton("🔧 Настроить предпочтения", callback_data='settings_change_prefs')],
        [InlineKeyboardButton("🔌 Деактивировать аккаунт", callback_data='settings_deactivate')],
        [InlineKeyboardButton("⬅️ Главное меню", callback_data='back_to_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_preferences_keyboard():
     """Клавиатура для настройки предпочтений."""
     keyboard = [
          # TODO: Добавить кнопки для изменения max_stocks, max_bonds, секторов
          [InlineKeyboardButton("🚧(в разработке) Макс. акций", callback_data='prefs_max_stocks')],
          [InlineKeyboardButton("🚧(в разработке) Макс. облигаций", callback_data='prefs_max_bonds')],
          [InlineKeyboardButton("🚧(в разработке) Исключить секторы", callback_data='prefs_exclude_sectors')],
          [InlineKeyboardButton("⬅️ Назад к настройкам", callback_data='settings_show')]
     ]
     return InlineKeyboardMarkup(keyboard)

def show_settings(update: Update, context: CallbackContext):
    """Показывает главное меню настроек."""
    query = update.callback_query
    if query: query.answer()
    user_tg = update.effective_user
    logger.info(f"User {user_tg.id} entered settings.")

    with get_db_session() as db:
        db_user = crud.get_user_by_telegram_id(db, user_tg.id)
        if not db_user:
             err_msg = "Ошибка: не могу найти ваш профиль. Попробуйте /start."
             if query: query.edit_message_text(err_msg)
             else: update.message.reply_text(err_msg)
             return ConversationHandler.END

        prefs = crud.get_user_preferences(db, db_user.id)

        profile_map = {
             "conservative": "Консервативный", "moderate": "Умеренный", "aggressive": "Агрессивный"
        }
        current_profile = profile_map.get(db_user.risk_profile, "Не определен")

        settings_text = f"⚙️ *Настройки*\n\n"
        settings_text += f"Ваш текущий риск-профиль: *{current_profile}*\n\n"

        if prefs:
             settings_text += "*Предпочтения портфеля:*\n"
             settings_text += f" • Максимум акций: {prefs.max_stocks}\n"
             settings_text += f" • Максимум облигаций: {prefs.max_bonds}\n"
             if prefs.excluded_sectors:
                  settings_text += f" • Исключенные секторы: {', '.join(prefs.excluded_sectors)}\n"
             if prefs.preferred_sectors:
                  settings_text += f" • Предпочтительные секторы: {', '.join(prefs.preferred_sectors)}\n"
        else:
             settings_text += "Предпочтения портфеля не заданы.\n"

        message_args = {
             "text": settings_text,
             "reply_markup": get_settings_keyboard(),
             "parse_mode": "Markdown"
        }

        if query:
             query.edit_message_text(**message_args)
        else:
             update.message.reply_text(**message_args)

        return SETTINGS_MAIN


def handle_settings_main_action(update: Update, context: CallbackContext):
     """Обрабатывает кнопки в главном меню настроек."""
     query = update.callback_query
     query.answer()
     data = query.data

     if data == 'settings_change_profile':
          query.edit_message_text(
               "Чтобы изменить риск-профиль, вам нужно будет снова пройти небольшой опрос.\n"
               "Хотите продолжить?",
               reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Да, пройти опрос", callback_data='profile_restart_yes')],
                    [InlineKeyboardButton("❌ Нет, вернуться", callback_data='settings_show')]
               ])
          )
          return SETTINGS_MAIN
     elif data == 'settings_change_prefs':
          query.edit_message_text(
               "🔧 *Настройка предпочтений*\n\nВыберите, что хотите изменить:",
               reply_markup=get_preferences_keyboard(),
               parse_mode="Markdown"
          )
          return SETTINGS_PREFERENCES
     elif data == 'settings_deactivate':
          query = update.callback_query
          query.answer()
          keyboard = [
               [InlineKeyboardButton("🔴 Да, деактивировать", callback_data='confirm_deactivate_yes')],
               [InlineKeyboardButton("🟢 Нет, оставить активным", callback_data='main_settings')] 
          ]
          query.edit_message_text(
               "⚠️ *Вы уверены, что хотите деактивировать аккаунт?*\n\n"
               "Вы перестанете получать еженедельные обновления портфеля.\n"
               "Ваши данные будут сохранены, и вы сможете вернуться, снова нажав /start.",
               reply_markup=InlineKeyboardMarkup(keyboard),
               parse_mode='Markdown'
          )
          return DEACTIVATE_CONFIRM 
     elif data == 'settings_show':
          return show_settings(update, context)
     elif data == 'profile_restart_yes':
          return start_risk_assessment(update, context)
     else:
          logger.warning(f"Unknown settings action: {data}")
          return SETTINGS_MAIN

def handle_settings_preferences_action(update: Update, context: CallbackContext):
     """Обрабатывает кнопки в меню настроек предпочтений."""
     query = update.callback_query
     query.answer("Эта функция пока не реализована")
     # TODO: Добавить логику изменения max_stocks, max_bonds, секторов
     # Потребуется ввод текста от пользователя (через MessageHandler)
     # или кнопки с вариантами.
     logger.warning(f"Unhandled preferences action: {query.data}")
     return SETTINGS_PREFERENCES

def get_settings_main_handlers():
    """Возвращает обработчики для состояния SETTINGS_MAIN."""
    return [
        CallbackQueryHandler(handle_settings_main_action, pattern='^settings_|^profile_restart_yes$'),
    ]

def get_settings_preferences_handlers():
     """Возвращает обработчики для состояния SETTINGS_PREFERENCES."""
     return [
          CallbackQueryHandler(handle_settings_preferences_action, pattern='^prefs_'),
          CallbackQueryHandler(show_settings, pattern='^settings_show$'),
     ]

