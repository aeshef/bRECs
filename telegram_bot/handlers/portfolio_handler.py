import logging
import matplotlib.pyplot as plt
import io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CallbackQueryHandler

# Настройка логирования
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
import sys
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from database.models import SessionLocal
from database import crud

# Состояния для ConversationHandler
PORTFOLIO_VIEW = 3
PORTFOLIO_REBALANCE = 4
START_ROUTES = 0

def show_portfolio(update: Update, context: CallbackContext):
    """
    Показывает текущий портфель пользователя
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
        
        # Получаем текущий портфель пользователя
        portfolio = crud.get_latest_portfolio(db, db_user.id)
        
        if not portfolio or not portfolio.weights:
            # Если у пользователя нет портфеля
            keyboard = [
                [InlineKeyboardButton("Создать портфель", callback_data="create_portfolio")],
                [InlineKeyboardButton("Назад", callback_data="back_to_main")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            query.edit_message_text(
                "У вас пока нет сгенерированного портфеля. "
                "Хотите создать новый портфель на основе вашего риск-профиля?",
                reply_markup=reply_markup
            )
            
            return PORTFOLIO_VIEW
        
        # Формируем сообщение с составом портфеля
        message = f"📊 *Ваш текущий инвестиционный портфель*\n\n"
        
        # Данные для построения графика
        labels = []
        sizes = []
        risk_free_total = 0
        stocks_total = 0
        
        for ticker, weight in portfolio.weights.items():
            weight_percent = weight * 100
            
            if ticker.startswith("OFZ") or ticker in ["CASH", "BOND", "RUOB"]:
                risk_free_total += weight
            else:
                stocks_total += weight
                
            if weight >= 0.02:  # Показываем только значимые доли (>= 2%)
                message += f"• {ticker}: {weight_percent:.1f}%\n"
                labels.append(ticker)
                sizes.append(weight)
        
        message += f"\n*Общая структура:*\n"
        message += f"• Безрисковые активы: {risk_free_total*100:.1f}%\n"
        message += f"• Акции: {stocks_total*100:.1f}%\n\n"
        
        message += f"Дата создания: {portfolio.created_at.strftime('%d.%m.%Y')}\n"
        message += f"Последнее обновление: {portfolio.updated_at.strftime('%d.%m.%Y')}\n"
        
        # Создаем диаграмму распределения активов
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Распределение активов в портфеле')
        
        # Сохраняем диаграмму в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Создаем кнопки
        keyboard = [
            [InlineKeyboardButton("🔄 Рекомендации по ребалансировке", callback_data="rebalance")],
            [InlineKeyboardButton("📈 Анализ эффективности", callback_data="performance")],
            [InlineKeyboardButton("Назад", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Сначала отправляем текстовое сообщение
        query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        # Затем отправляем диаграмму
        context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=buf,
            caption="Распределение активов в вашем портфеле"
        )
        
        return PORTFOLIO_VIEW
    
    except Exception as e:
        logger.error(f"Ошибка при отображении портфеля: {e}")
        query.edit_message_text(
            "Произошла ошибка при загрузке портфеля. Пожалуйста, попробуйте позже."
        )
        return START_ROUTES
    
    finally:
        db.close()

def show_rebalance(update: Update, context: CallbackContext):
    """
    Показывает рекомендации по ребалансировке портфеля
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
        
        # Получаем текущий портфель пользователя
        current_portfolio = crud.get_latest_portfolio(db, db_user.id)
        
        if not current_portfolio or not current_portfolio.weights:
            # Если у пользователя нет портфеля
            keyboard = [
                [InlineKeyboardButton("Создать портфель", callback_data="create_portfolio")],
                [InlineKeyboardButton("Назад", callback_data="back_to_main")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            query.edit_message_text(
                "У вас пока нет сгенерированного портфеля. "
                "Хотите создать новый портфель на основе вашего риск-профиля?",
                reply_markup=reply_markup
            )
            
            return PORTFOLIO_VIEW
        
        # Здесь должно быть получение рекомендаций по ребалансировке
        # В реальном проекте здесь будет вызов вашей функции для получения рекомендаций
        # Сейчас используем заглушку
        
        # Пример рекомендаций:
        rebalance_recommendations = [
            {"ticker": "SBER", "current_weight": 0.15, "target_weight": 0.18, "action": "BUY", "change_pct": 0.03},
            {"ticker": "GAZP", "current_weight": 0.12, "target_weight": 0.10, "action": "SELL", "change_pct": 0.02},
            {"ticker": "OFZ-26235", "current_weight": 0.25, "target_weight": 0.22, "action": "SELL", "change_pct": 0.03}
        ]
        
        if not rebalance_recommendations:
            query.edit_message_text(
                "✅ Ваш портфель не требует ребалансировки в настоящий момент.\n\n"
                "Рекомендуем проверить необходимость ребалансировки через неделю "
                "или при значительных изменениях на рынке.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Назад к портфелю", callback_data="portfolio")],
                    [InlineKeyboardButton("Главное меню", callback_data="back_to_main")]
                ])
            )
            return PORTFOLIO_REBALANCE
        
        # Формируем сообщение с рекомендациями по ребалансировке
        message = "🔄 *Рекомендации по ребалансировке портфеля*\n\n"
        
        for rec in rebalance_recommendations:
            action = "🟢 Купить" if rec['action'] == 'BUY' else "🔴 Продать"
            message += f"{action} {rec['ticker']}: {rec['current_weight']*100:.1f}% → {rec['target_weight']*100:.1f}% "
            message += f"({abs(rec['change_pct']*100):.1f}%)\n"
        
        message += "\n*Примечание:* Рекомендации основаны на текущей рыночной ситуации и целевом распределении портфеля."
        
        keyboard = [
            [InlineKeyboardButton("📊 Вернуться к портфелю", callback_data="portfolio")],
            [InlineKeyboardButton("Назад", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        return PORTFOLIO_REBALANCE
    
    except Exception as e:
        logger.error(f"Ошибка при отображении рекомендаций по ребалансировке: {e}")
        query.edit_message_text(
            "Произошла ошибка при загрузке рекомендаций. Пожалуйста, попробуйте позже."
        )
        return START_ROUTES
    
    finally:
        db.close()

def handle_portfolio_action(update: Update, context: CallbackContext):
    """
    Обрабатывает действия в разделе портфеля
    """
    query = update.callback_query
    query.answer()
    
    if query.data == "create_portfolio":
        # Создание нового портфеля
        query.edit_message_text(
            "🔄 Генерируем для вас оптимальный портфель...\n\n"
            "Это может занять несколько минут. Мы учитываем текущую рыночную ситуацию, "
            "фундаментальные показатели и технический анализ."
        )
        
        # Здесь должен быть вызов функции для генерации портфеля
        # Это заглушка, которую нужно заменить реальным вызовом
        import time
        time.sleep(2)  # Имитация длительного процесса
        
        # После создания портфеля показываем его
        return show_portfolio(update, context)
    
    elif query.data == "performance":
        # Показ анализа эффективности
        query.edit_message_text(
            "📈 *Анализ эффективности портфеля*\n\n"
            "Функция находится в разработке и будет доступна в ближайшее время.\n\n"
            "Здесь вы сможете увидеть динамику стоимости портфеля, сравнение с бенчмарком, "
            "основные метрики риска и доходности.",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Назад к портфелю", callback_data="portfolio")],
                [InlineKeyboardButton("Главное меню", callback_data="back_to_main")]
            ])
        )
        return PORTFOLIO_VIEW
    
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

def get_view_handlers():
    """
    Возвращает список обработчиков для состояния PORTFOLIO_VIEW
    """
    return [
        CallbackQueryHandler(handle_portfolio_action)
    ]

def get_rebalance_handlers():
    """
    Возвращает список обработчиков для состояния PORTFOLIO_REBALANCE
    """
    return [
        CallbackQueryHandler(handle_portfolio_action)
    ]
