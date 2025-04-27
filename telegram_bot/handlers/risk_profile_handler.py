import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, MessageHandler, Filters, CallbackQueryHandler

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
import sys
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from database.models import SessionLocal
from database import crud

# Состояния для ConversationHandler
RISK_ASSESSMENT = 1
RISK_PROFILE_SETUP = 2

# Вопросы для определения риск-профиля
RISK_QUESTIONS = [
    {
        "question": "Какой горизонт инвестирования вы рассматриваете?",
        "options": [
            {"text": "До 1 года", "score": 1},
            {"text": "1-3 года", "score": 2},
            {"text": "3-5 лет", "score": 3},
            {"text": "Более 5 лет", "score": 4}
        ]
    },
    {
        "question": "Как вы отреагируете на падение стоимости вашего портфеля на 20%?",
        "options": [
            {"text": "Продам все активы немедленно", "score": 1},
            {"text": "Продам часть активов", "score": 2},
            {"text": "Ничего не буду делать", "score": 3},
            {"text": "Докуплю активов", "score": 4}
        ]
    },
    {
        "question": "Какая доходность вас интересует?",
        "options": [
            {"text": "Стабильная, немного выше депозита", "score": 1},
            {"text": "Умеренная, 10-15% годовых", "score": 2},
            {"text": "Высокая, 15-25% годовых", "score": 3},
            {"text": "Максимальная, готов к рискам", "score": 4}
        ]
    },
    {
        "question": "Какой опыт инвестирования у вас есть?",
        "options": [
            {"text": "Нет опыта", "score": 1},
            {"text": "Есть опыт с депозитами/ОФЗ", "score": 2},
            {"text": "Торговал акциями/ETF", "score": 3},
            {"text": "Большой опыт, включая сложные инструменты", "score": 4}
        ]
    },
    {
        "question": "Какую часть свободных средств вы готовы инвестировать?",
        "options": [
            {"text": "До 10%", "score": 1},
            {"text": "10-30%", "score": 2},
            {"text": "30-60%", "score": 3},
            {"text": "Более 60%", "score": 4}
        ]
    }
]

def start_risk_assessment(update: Update, context: CallbackContext):
    """
    Начинает процесс оценки риск-профиля пользователя
    """
    # Инициализируем список для хранения ответов
    context.user_data['risk_answers'] = []
    
    # Отправляем первый вопрос
    return ask_question(update, context, 0)

def ask_question(update: Update, context: CallbackContext, question_index):
    """
    Задает вопрос для определения риск-профиля
    """
    if question_index >= len(RISK_QUESTIONS):
        # Если все вопросы заданы, переходим к анализу результатов
        return analyze_risk_profile(update, context)
    
    question = RISK_QUESTIONS[question_index]
    
    # Создаем кнопки с вариантами ответов
    keyboard = []
    for i, option in enumerate(question["options"]):
        keyboard.append([InlineKeyboardButton(option["text"], callback_data=f"risk_{question_index}_{i}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Определяем, использовать update.message или update.callback_query
    if update.callback_query:
        update.callback_query.edit_message_text(
            f"Вопрос {question_index + 1}/{len(RISK_QUESTIONS)}:\n\n{question['question']}",
            reply_markup=reply_markup
        )
    else:
        update.message.reply_text(
            f"Вопрос {question_index + 1}/{len(RISK_QUESTIONS)}:\n\n{question['question']}",
            reply_markup=reply_markup
        )
    
    return RISK_ASSESSMENT

def handle_answer(update: Update, context: CallbackContext):
    """
    Обрабатывает ответ на вопрос
    """
    query = update.callback_query
    query.answer()
    
    # Извлекаем индекс вопроса и ответа
    _, question_index, option_index = query.data.split("_")
    question_index = int(question_index)
    option_index = int(option_index)
    
    # Сохраняем ответ
    context.user_data['risk_answers'].append({
        'question_index': question_index,
        'option_index': option_index,
        'score': RISK_QUESTIONS[question_index]["options"][option_index]["score"]
    })
    
    # Переходим к следующему вопросу
    return ask_question(update, context, question_index + 1)

def analyze_risk_profile(update: Update, context: CallbackContext):
    """
    Анализирует ответы и определяет риск-профиль пользователя
    """
    # Вычисляем общий балл
    total_score = sum(answer['score'] for answer in context.user_data.get('risk_answers', []))
    max_possible_score = len(RISK_QUESTIONS) * 4  # Максимальный балл за каждый вопрос - 4
    
    # Определяем риск-профиль на основе процента от максимального балла
    score_percentage = total_score / max_possible_score if max_possible_score > 0 else 0
    
    if score_percentage < 0.4:
        risk_profile = "conservative"
        profile_description = "Консервативный инвестор предпочитает стабильность и минимальный риск. Портфель будет включать больше облигаций и меньше акций."
    elif score_percentage < 0.7:
        risk_profile = "moderate"
        profile_description = "Умеренный инвестор готов на некоторый риск ради более высокой доходности. Портфель будет сбалансирован между акциями и облигациями."
    else:
        risk_profile = "aggressive"
        profile_description = "Агрессивный инвестор нацелен на максимальную доходность и готов к высоким рискам. Портфель будет преимущественно состоять из акций."
    
    # Сохраняем риск-профиль в базе данных
    user = update.effective_user
    db = SessionLocal()
    
    try:
        crud.update_user_risk_profile(db, user.id, risk_profile)
        
        # Создаем стандартные предпочтения пользователя
        if risk_profile == "conservative":
            crud.create_or_update_preferences(
                db, 
                user_id=crud.get_user_by_telegram_id(db, user.id).id,
                max_stocks=5,
                max_bonds=10
            )
        elif risk_profile == "moderate":
            crud.create_or_update_preferences(
                db, 
                user_id=crud.get_user_by_telegram_id(db, user.id).id,
                max_stocks=10,
                max_bonds=5
            )
        else:  # aggressive
            crud.create_or_update_preferences(
                db, 
                user_id=crud.get_user_by_telegram_id(db, user.id).id,
                max_stocks=15,
                max_bonds=3
            )
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении риск-профиля: {e}")
    
    finally:
        db.close()
    
    # Отправляем результат пользователю
    keyboard = [
        [InlineKeyboardButton("Перейти к настройкам портфеля", callback_data="setup_portfolio")],
        [InlineKeyboardButton("Вернуться в главное меню", callback_data="back_to_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.callback_query.edit_message_text(
        f"Ваш инвестиционный профиль: *{get_profile_name_russian(risk_profile)}*\n\n"
        f"{profile_description}\n\n"
        f"Общий балл: {total_score} из {max_possible_score}",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    
    return RISK_PROFILE_SETUP

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

def handle_risk_profile_setup(update: Update, context: CallbackContext):
    """
    Обрабатывает действия после определения риск-профиля
    """
    query = update.callback_query
    query.answer()
    
    if query.data == "setup_portfolio":
        # Генерируем начальный портфель на основе риск-профиля
        
        # Временное сообщение о генерации портфеля
        query.edit_message_text(
            "🔄 Генерируем для вас оптимальный портфель на основе вашего риск-профиля...\n\n"
            "Это может занять несколько минут. Мы учитываем текущую рыночную ситуацию, "
            "фундаментальные показатели и технический анализ для создания наилучшего портфеля."
        )
        
        # Здесь должен быть вызов функции для генерации портфеля
        # Это заглушка, которую нужно заменить реальным вызовом
        import time
        time.sleep(2)  # Имитация длительного процесса
        
        keyboard = [
            [InlineKeyboardButton("📊 Посмотреть портфель", callback_data="portfolio")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            "✅ Портфель успешно сгенерирован!\n\n"
            "Теперь вы можете просмотреть рекомендованный состав портфеля "
            "или настроить дополнительные параметры.",
            reply_markup=reply_markup
        )
        
        return START_ROUTES
    
    elif query.data == "back_to_main":
        # Возвращаемся в главное меню
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

def get_handlers():
    """
    Возвращает список обработчиков для состояния RISK_ASSESSMENT
    """
    return [
        CallbackQueryHandler(handle_answer, pattern=r"^risk_\d+_\d+$")
    ]

def get_setup_handlers():
    """
    Возвращает список обработчиков для состояния RISK_PROFILE_SETUP
    """
    return [
        CallbackQueryHandler(handle_risk_profile_setup)
    ]
