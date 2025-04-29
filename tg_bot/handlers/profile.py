import logging
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CallbackQueryHandler, ConversationHandler

from db import crud, get_db_session
from tg_bot.states import START_ROUTES, RISK_ASSESSMENT, RISK_PROFILE_SETUP
from tg_bot.async_tasks import run_pipeline_and_notify_user


logger = logging.getLogger(__name__)

RISK_QUESTIONS = [
    {
        "question": "Какой горизонт инвестирования вы рассматриваете?",
        "options": [
            {"text": "До 1 года", "score": 1, "data": "horizon_1"},
            {"text": "1-3 года", "score": 2, "data": "horizon_3"},
            {"text": "3-5 лет", "score": 3, "data": "horizon_5"},
            {"text": "Более 5 лет", "score": 4, "data": "horizon_long"}
        ]
    },
    {
        "question": "Как вы отреагируете на падение стоимости вашего портфеля на 20%?",
        "options": [
            {"text": "Продам все активы", "score": 1, "data": "react_sell_all"},
            {"text": "Продам часть активов", "score": 2, "data": "react_sell_part"},
            {"text": "Ничего не буду делать", "score": 3, "data": "react_hold"},
            {"text": "Докуплю активов", "score": 4, "data": "react_buy_more"}
        ]
    },
    {
        "question": "Какая доходность вас интересует?",
        "options": [
            {"text": "Стабильная (как депозит)", "score": 1, "data": "yield_low"},
            {"text": "Умеренная (10-15% годовых)", "score": 2, "data": "yield_mid"},
            {"text": "Высокая (15-25% годовых)", "score": 3, "data": "yield_high"},
            {"text": "Максимальная (риск не важен)", "score": 4, "data": "yield_max"}
        ]
    },
    {
        "question": "Ваш опыт инвестирования?",
        "options": [
            {"text": "Нет опыта", "score": 1, "data": "exp_none"},
            {"text": "Депозиты/Облигации", "score": 2, "data": "exp_basic"},
            {"text": "Акции/ETF", "score": 3, "data": "exp_stock"},
            {"text": "Профи (фьючерсы и т.д.)", "score": 4, "data": "exp_pro"}
        ]
    },
    {
        "question": "Какую долю капитала готовы инвестировать?",
        "options": [
            {"text": "До 10%", "score": 1, "data": "capital_10"},
            {"text": "10-30%", "score": 2, "data": "capital_30"},
            {"text": "30-60%", "score": 3, "data": "capital_60"},
            {"text": "Более 60%", "score": 4, "data": "capital_over"}
        ]
    }
]

RISK_CALLBACK_PREFIX = "riskanswer_"

# --- Функции Обработчиков ---
def start_risk_assessment(update: Update, context: CallbackContext):
    """Начинает опрос для определения риск-профиля."""
    context.user_data['risk_answers'] = []
    context.user_data['current_question'] = 0
    return ask_next_question(update, context)

def ask_next_question(update: Update, context: CallbackContext) -> int:
    """Задает текущий вопрос или завершает опрос."""
    question_index = context.user_data.get('current_question', 0)

    if question_index >= len(RISK_QUESTIONS):
        return analyze_risk_profile(update, context)

    question_data = RISK_QUESTIONS[question_index]
    text = f"❓ *Вопрос {question_index + 1} из {len(RISK_QUESTIONS)}*\n\n{question_data['question']}"
    keyboard = []
    for option in question_data['options']:
        callback_data = f"{RISK_CALLBACK_PREFIX}{question_index}_{option['data']}"
        keyboard.append([InlineKeyboardButton(option['text'], callback_data=callback_data)])

    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        try:
             update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode='Markdown')
        except Exception as e:
             logger.warning(f"Could not edit message: {e}. Sending new one.")
             update.effective_chat.send_message(text, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        update.message.reply_text(text, reply_markup=reply_markup, parse_mode='Markdown')

    return RISK_ASSESSMENT

def handle_risk_answer(update: Update, context: CallbackContext):
    """Обрабатывает ответ на вопрос опросника."""
    query = update.callback_query
    query.answer()
    callback_data = query.data

    try:
        parts = callback_data.replace(RISK_CALLBACK_PREFIX, "").split("_", 1)
        question_index = int(parts[0])
        answer_data = parts[1]
    except (IndexError, ValueError):
        logger.error(f"Invalid risk answer callback data: {callback_data}")
        query.message.reply_text("Произошла ошибка. Попробуйте еще раз.")
        return ask_next_question(update, context)

    score = 0
    question_options = RISK_QUESTIONS[question_index]['options']
    for option in question_options:
        if option['data'] == answer_data:
            score = option['score']
            break

    if 'risk_answers' not in context.user_data: context.user_data['risk_answers'] = []
    context.user_data['risk_answers'].append({'question': question_index, 'score': score})
    context.user_data['current_question'] = question_index + 1

    return ask_next_question(update, context)

def analyze_risk_profile(update: Update, context: CallbackContext) -> int:
    """Анализирует ответы, сохраняет профиль и запускает генерацию портфеля."""
    user_tg = update.effective_user
    answers = context.user_data.get('risk_answers', [])
    if not answers:
        logger.warning(f"No answers found for user {user_tg.id} in analyze_risk_profile")
        return start_risk_assessment(update, context)

    total_score = sum(a['score'] for a in answers)
    max_score = sum(max(opt['score'] for opt in q['options']) for q in RISK_QUESTIONS)
    score_percentage = total_score / max_score if max_score > 0 else 0

    if score_percentage < 0.4:
        risk_profile = "conservative"
        profile_name = "Консервативный"
        profile_desc = "Вы предпочитаете минимальный риск и стабильность."
    elif score_percentage < 0.7:
        risk_profile = "moderate"
        profile_name = "Умеренный"
        profile_desc = "Вы готовы на умеренный риск ради большей доходности."
    else:
        risk_profile = "aggressive"
        profile_name = "Агрессивный"
        profile_desc = "Вы нацелены на максимальную доходность и готовы к высоким рискам."

    logger.info(f"User {user_tg.id} risk profile determined as '{risk_profile}' (Score: {total_score}/{max_score})")

    with get_db_session() as db:
        try:
            db_user = crud.update_user_risk_profile(db, user_tg.id, risk_profile)
            if not db_user:
                 raise Exception("User not found in DB during profile saving.")
            user_id_db = db_user.id
            logger.info(f"Saved risk profile for user_id {user_id_db}")

            # TODO: Можно обновить/установить UserPreferences по умолчанию на основе профиля
            # crud.create_or_update_preferences(db, user_id_db, max_stocks=..., max_bonds=...)

        except Exception as e:
            logger.exception(f"Failed to save risk profile for user {user_tg.id}: {e}")
            update.effective_chat.send_message("Не удалось сохранить ваш профиль. Пожалуйста, попробуйте позже.")
            return ConversationHandler.END

    result_text = (
        f"✅ Спасибо за ответы!\n\n"
        f"Ваш инвестиционный профиль: *{profile_name}*\n"
        f"_{profile_desc}_\n\n"
        f"🤖 Сейчас я подберу для вас стартовый портфель. Это может занять несколько минут..."
    )

    message_to_edit = update.callback_query.message if update.callback_query else update.message
    message_to_edit.edit_text(result_text, reply_markup=None, parse_mode='Markdown')

    logger.info(f"Dispatching async task 'run_pipeline_and_notify_user' for user_id {user_id_db}")
    context.dispatcher.run_async(
        run_pipeline_and_notify_user,
        user_id_db=user_id_db,
        chat_id=update.effective_chat.id,
        is_initial=True
    )

    context.user_data.pop('risk_answers', None)
    context.user_data.pop('current_question', None)

    return ConversationHandler.END

def get_risk_assessment_handlers():
    """Возвращает обработчики для состояния RISK_ASSESSMENT."""
    return [
        CallbackQueryHandler(handle_risk_answer, pattern=f'^{RISK_CALLBACK_PREFIX}')
    ]

def get_risk_profile_setup_handlers():
     """Обработчики для состояния RISK_PROFILE_SETUP (если оно нужно)."""
     # В данном коде анализ и запуск происходят сразу после последнего ответа,
     # поэтому отдельное состояние RISK_PROFILE_SETUP не используется.
     # Если бы мы показывали профиль и спрашивали "Согласны?", то хендлеры были бы здесь.
     return []
