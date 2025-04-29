import logging
import json
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (CallbackContext, CommandHandler, CallbackQueryHandler,
                          ConversationHandler, MessageHandler, Filters)

from db import crud, get_db_session
from ..states import (START_ROUTES, RISK_ASSESSMENT, PORTFOLIO_VIEW, SETTINGS_MAIN,
                      CONTACT_ADMIN, DEACTIVATE_CONFIRM)

from ..utils.keyboard_utils import get_main_keyboard
from . import profile, portfolio, settings


logger = logging.getLogger(__name__)

# Загрузка ID админов
try:
    ADMIN_IDS = json.loads(os.getenv('TELEGRAM_ADMIN_IDS', '[]'))
    if not isinstance(ADMIN_IDS, list): ADMIN_IDS = []
except Exception:
    ADMIN_IDS = []

# --- Обработчики ---
def start(update: Update, context: CallbackContext):
    """Обработчик команды /start."""
    user_tg = update.effective_user
    logger.info(f"User {user_tg.id} started bot.")

    with get_db_session() as db:
        db_user = crud.get_user_by_telegram_id(db, user_tg.id)

        if not db_user:
            # Новый пользователь
            logger.info(f"Creating new user for telegram_id {user_tg.id}")
            try:
                db_user = crud.create_user(
                    db,
                    telegram_id=user_tg.id,
                    username=user_tg.username,
                    first_name=user_tg.first_name,
                    last_name=user_tg.last_name
                )
                update.message.reply_text(
                    f"Добро пожаловать, {user_tg.first_name}! 👋\n\n"
                    "Я помогу вам сформировать инвестиционный портфель.\n"
                    "Для начала, давайте определим ваш инвестиционный профиль."
                )
                # Переходим к определению риск-профиля (из profile.py)
                return profile.start_risk_assessment(update, context)
            except Exception as e:
                logger.exception(f"Error creating user {user_tg.id}: {e}")
                update.message.reply_text("Произошла ошибка при регистрации. Пожалуйста, попробуйте /start еще раз.")
                return ConversationHandler.END

        elif not db_user.risk_profile:
             # Пользователь есть, но профиль не заполнен (например, прервал раньше)
             logger.info(f"User {user_tg.id} exists but has no profile. Starting assessment.")
             update.message.reply_text(
                 f"С возвращением, {user_tg.first_name}! Похоже, вы не завершили определение риск-профиля. Давайте продолжим."
             )
             return profile.start_risk_assessment(update, context)

        elif not db_user.is_active:
             # Пользователь был деактивирован и вернулся
             logger.info(f"Re-activating user {user_tg.id}")
             crud.update_user_active_status(db, user_tg.id, True)
             update.message.reply_text(
                 f"С возвращением, {user_tg.first_name}! Ваш аккаунт снова активен.",
                 reply_markup=get_main_keyboard()
             )
             return START_ROUTES

        else:
            # Существующий активный пользователь
            logger.info(f"Existing user {user_tg.id} returned.")
            update.message.reply_text(
                f"С возвращением, {user_tg.first_name}! 👋",
                reply_markup=get_main_keyboard()
            )
            return START_ROUTES

def main_menu_handler(update: Update, context: CallbackContext):
    """Обработчик кнопок главного меню."""
    query = update.callback_query
    query.answer()
    data = query.data

    if data == 'main_portfolio':
        return portfolio.show_portfolio(update, context) # Переход в PORTFOLIO_VIEW
    elif data == 'main_settings':
        return settings.show_settings(update, context) # Переход в SETTINGS_MAIN
    elif data == 'main_help':
        query.edit_message_text(get_help_text(), parse_mode='Markdown')
        query.edit_message_reply_markup(reply_markup=get_main_keyboard())
        return START_ROUTES
    else:
         logger.warning(f"Unknown main menu callback data: {data}")
         query.edit_message_text("Неизвестная команда.", reply_markup=get_main_keyboard())
         return START_ROUTES

def back_to_main_menu(update: Update, context: CallbackContext):
     """Возврат в главное меню из других разделов."""
     query = update.callback_query
     query.answer()
     query.edit_message_text("Главное меню:", reply_markup=get_main_keyboard())
     return START_ROUTES

def get_help_text():
     """Возвращает текст справки."""
     return (
         "📚 *Справка*\n\n"
         "• *Мой портфель* - посмотреть текущий рекомендованный состав портфеля.\n"
         "• *Настройки* - изменить риск-профиль или другие параметры.\n"
         "• `/start` - вернуться в главное меню.\n"
         "• `/help` - показать эту справку.\n"
         "• `/cancel` - отменить текущее действие (например, опрос).\n"
         "• `/contact` - отправить сообщение администратору.\n\n"
         "Бот будет периодически (раз в неделю) пересчитывать ваш портфель и присылать уведомления, если потребуются изменения."
     )

def help_command(update: Update, context: CallbackContext):
    """Обработчик команды /help."""
    update.message.reply_text(get_help_text(), parse_mode='Markdown', reply_markup=get_main_keyboard())
    return START_ROUTES

def contact_command(update: Update, context: CallbackContext):
     """Начинает диалог для отправки сообщения админу."""
     update.message.reply_text(
         "Введите ваше сообщение для администратора. Оно будет переслано ему."
         "Для отмены введите /cancel."
     )
     return CONTACT_ADMIN

def forward_to_admin(update: Update, context: CallbackContext):
     """Пересылает сообщение пользователя админу."""
     user = update.effective_user
     message_text = update.message.text
     logger.info(f"User {user.id} sending message to admin.")

     if not ADMIN_IDS:
          update.message.reply_text("К сожалению, связь с администратором сейчас недоступна.")
          return ConversationHandler.END

     forward_message = (
         f"Сообщение от пользователя:\n"
         f"ID: {user.id}\n"
         f"Имя: {user.full_name}\n"
         f"Username: @{user.username or 'нет'}\n\n"
         f"Сообщение:\n---\n{message_text}\n---"
     )

     sent_count = 0
     for admin_id in ADMIN_IDS:
          try:
               context.bot.send_message(chat_id=admin_id, text=forward_message)
               sent_count += 1
          except Exception as e:
               logger.error(f"Failed to forward message to admin {admin_id}: {e}")

     if sent_count > 0:
          update.message.reply_text("Ваше сообщение успешно отправлено администратору!")
     else:
          update.message.reply_text("Не удалось отправить сообщение администратору. Попробуйте позже.")

     update.message.reply_text("Главное меню:", reply_markup=get_main_keyboard())
     return START_ROUTES

def request_deactivate_confirm(update: Update, context: CallbackContext):
     """Запрашивает подтверждение деактивации."""
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

def deactivate_user(update: Update, context: CallbackContext):
     """Деактивирует пользователя."""
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
               query.edit_message_reply_markup(reply_markup=settings.get_settings_keyboard())
               return SETTINGS_MAIN

def cancel(update: Update, context: CallbackContext):
    """Отмена текущего диалога и возврат в главное меню."""
    user = update.effective_user
    logger.info(f"User {user.id} canceled operation.")
    update.message.reply_text(
        'Действие отменено.', reply_markup=get_main_keyboard()
    )
    return START_ROUTES

def error_handler(update: object, context: CallbackContext) -> None:
    """Логирует ошибки и отправляет сообщение пользователю."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    if isinstance(update, Update) and update.effective_chat:
         try:
              context.bot.send_message(
                  chat_id=update.effective_chat.id,
                  text="Произошла внутренняя ошибка. Пожалуйста, попробуйте позже или свяжитесь с поддержкой."
              )
         except Exception as e:
              logger.error(f"Exception while sending error message to user: {e}")


def get_start_routes_handlers():
    """Возвращает обработчики для состояния START_ROUTES."""
    return [
        CallbackQueryHandler(main_menu_handler, pattern='^main_'),
        CallbackQueryHandler(back_to_main_menu, pattern='^back_to_main$'),
    ]

def get_contact_admin_handlers():
     """Возвращает обработчики для состояния CONTACT_ADMIN."""
     return [MessageHandler(Filters.text & ~Filters.command, forward_to_admin)]

def get_deactivate_confirm_handlers():
     """Возвращает обработчики для состояния DEACTIVATE_CONFIRM."""
     return [
          CallbackQueryHandler(deactivate_user, pattern='^confirm_deactivate_yes$'),
          CallbackQueryHandler(settings.show_settings, pattern='^main_settings$')
     ]
