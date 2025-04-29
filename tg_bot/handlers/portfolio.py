import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CallbackQueryHandler, ConversationHandler

from db import crud, get_db_session
from tg_bot.notification_sender import format_portfolio_message, create_portfolio_pie_chart
from tg_bot.states import PORTFOLIO_VIEW, START_ROUTES
from ..utils.keyboard_utils import get_main_keyboard

logger = logging.getLogger(__name__)

def show_portfolio(update: Update, context: CallbackContext):
    """Показывает текущий (последний) портфель пользователя."""
    query = update.callback_query
    if query: query.answer()

    user_tg = update.effective_user
    logger.info(f"User {user_tg.id} requested to view portfolio.")

    with get_db_session() as db:
        db_user = crud.get_user_by_telegram_id(db, user_tg.id)
        if not db_user:
             err_msg = "Ошибка: не могу найти ваш профиль. Попробуйте /start."
             if query: query.edit_message_text(err_msg)
             else: update.message.reply_text(err_msg)
             return ConversationHandler.END

        # Получаем самый свежий портфель
        latest_portfolio = crud.get_latest_portfolio(db, db_user.id)

        if not latest_portfolio or not latest_portfolio.weights:
             no_portfolio_text = ("У вас пока нет готового портфеля.\n"
                                  "Я сообщу, как только он будет рассчитан.")
             if db_user.risk_profile:
                  no_portfolio_text += "\nОбычно это занимает несколько минут после определения риск-профиля."

             keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data='back_to_main')]]
             reply_markup = InlineKeyboardMarkup(keyboard)

             if query: query.edit_message_text(no_portfolio_text, reply_markup=reply_markup)
             else: update.message.reply_text(no_portfolio_text, reply_markup=reply_markup)
             return PORTFOLIO_VIEW

        # --- Отображаем портфель ---
        try:
             # Формируем текст сообщения
             message_text = format_portfolio_message(
                 weights=latest_portfolio.weights,
                 metrics=latest_portfolio.metrics,
                 is_initial=False,
                 significant_changes=False
             )
             message_text = f"📊 *Ваш текущий портфель*\n_(от {latest_portfolio.created_at.strftime('%d.%m.%Y %H:%M')})_\n\n" + message_text

             chart_buf = create_portfolio_pie_chart(latest_portfolio.weights)

             keyboard = [
                  # [InlineKeyboardButton("🔄 Проверить обновления", callback_data='portfolio_check_update')], # Кнопка для ручного запуска? Пока не делаем
                  [InlineKeyboardButton("⚙️ Настройки", callback_data='main_settings')],
                  [InlineKeyboardButton("⬅️ Главное меню", callback_data='back_to_main')]
             ]
             reply_markup = InlineKeyboardMarkup(keyboard)

             if query:
                 query.edit_message_text(message_text, parse_mode='Markdown', reply_markup=reply_markup)
                 if chart_buf:
                      context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_buf)
             else:
                  if chart_buf:
                       context.bot.send_photo(chat_id=update.message.chat_id, photo=chart_buf,
                                              caption=message_text, parse_mode='Markdown', reply_markup=reply_markup)
                  else:
                       update.message.reply_text(message_text, parse_mode='Markdown', reply_markup=reply_markup)

             return PORTFOLIO_VIEW

        except Exception as e:
             logger.exception(f"Error formatting/sending portfolio for user {user_tg.id}: {e}")
             err_msg = "Произошла ошибка при отображении портфеля."
             if query: query.edit_message_text(err_msg, reply_markup=get_main_keyboard())
             else: update.message.reply_text(err_msg, reply_markup=get_main_keyboard())
             return START_ROUTES


def handle_portfolio_view_buttons(update: Update, context: CallbackContext):
     """Обработчик кнопок в состоянии PORTFOLIO_VIEW."""
     # Сейчас тут нет специфичных кнопок, кроме общих (Назад, Настройки),
     # которые обрабатываются в common.py и settings.py по их callback_data.
     # Если добавить кнопки (например, "Детали по активу"), их обработчики будут здесь.
     query = update.callback_query
     query.answer("Это действие пока не реализовано")
     logger.warning(f"Unhandled button in PORTFOLIO_VIEW: {query.data}")
     return PORTFOLIO_VIEW # Остаемся здесь


def get_portfolio_view_handlers():
    """Возвращает обработчики для состояния PORTFOLIO_VIEW."""
    # Основное действие - показать портфель - вызывается извне (из main_menu_handler)
    # Здесь могут быть обработчики кнопок *под* портфелем, если они специфичны для этого состояния
    return [
         CallbackQueryHandler(handle_portfolio_view_buttons)
    ]
