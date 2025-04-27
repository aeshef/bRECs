import logging
import json
from telegram import Bot, ParseMode
import io
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/portfolio-advisor/logs/notifications.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импортируем конфигурацию бота
import sys
sys.path.append('/opt/portfolio-advisor')
from config.bot_config import BOT_CONFIG

# Инициализируем бота
bot = Bot(token=BOT_CONFIG['token'])

def send_portfolio_update(telegram_id, recommendations, is_urgent=False):
    """
    Отправляет уведомление о обновлении портфеля
    
    Args:
        telegram_id: ID пользователя в Telegram
        recommendations: Словарь с рекомендациями
        is_urgent: Флаг срочности уведомления
    """
    try:
        # Формируем сообщение
        if is_urgent:
            message = "🚨 *СРОЧНОЕ УВЕДОМЛЕНИЕ*\n\n"
        else:
            message = "📊 *Обновление инвестиционного портфеля*\n\n"
        
        if 'recommendation' in recommendations:
            message += f"{recommendations['recommendation']}\n\n"
        
        if 'reason' in recommendations:
            message += f"*Причина:* {recommendations['reason']}\n\n"
        
        # Добавляем информацию о портфеле
        if 'weights' in recommendations and recommendations['weights']:
            message += "*Рекомендуемый состав портфеля:*\n"
            
            # Сортируем веса по убыванию для лучшей читаемости
            sorted_weights = sorted(
                recommendations['weights'].items(), 
                key=lambda item: item[1], 
                reverse=True
            )
            
            # Данные для графика
            labels = []
            sizes = []
            
            for ticker, weight in sorted_weights:
                if weight >= 0.02:  # Показываем только значимые доли (>= 2%)
                    message += f"• {ticker}: {weight*100:.1f}%\n"
                    labels.append(ticker)
                    sizes.append(weight)
        
        # Добавляем метрики портфеля, если они есть
        if 'metrics' in recommendations and recommendations['metrics']:
            metrics = recommendations['metrics']
            message += "\n*Ожидаемые характеристики портфеля:*\n"
            
            if 'expected_return' in metrics:
                message += f"• Ожидаемая доходность: {metrics['expected_return']*100:.2f}%\n"
            
            if 'volatility' in metrics:
                message += f"• Волатильность: {metrics['volatility']*100:.2f}%\n"
            
            if 'sharpe_ratio' in metrics:
                message += f"• Коэффициент Шарпа: {metrics['sharpe_ratio']:.2f}\n"
        
        # Создаем график распределения портфеля, если есть достаточно данных
        if len(labels) > 0 and len(sizes) > 0:
            plt.figure(figsize=(10, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Рекомендуемое распределение активов')
            
            # Сохраняем график в буфер
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Сначала отправляем текстовое сообщение
            bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Затем отправляем график
            bot.send_photo(
                chat_id=telegram_id,
                photo=buf,
                caption="Рекомендуемое распределение активов в портфеле"
            )
        else:
            # Если нет данных для графика, отправляем только текст
            bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        
        logger.info(f"Уведомление об обновлении портфеля отправлено пользователю {telegram_id}")
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при отправке уведомления пользователю {telegram_id}: {e}")
        return False

def send_urgent_signal(telegram_id, urgent_recommendations):
    """
    Отправляет экстренное уведомление о необходимости изменения портфеля
    
    Args:
        telegram_id: ID пользователя в Telegram
        urgent_recommendations: Словарь с экстренными рекомендациями
    """
    # Используем ту же функцию, но с флагом срочности
    return send_portfolio_update(telegram_id, urgent_recommendations, is_urgent=True)

def broadcast_message(message, user_ids=None):
    """
    Отправляет сообщение всем указанным пользователям или всем активным
    
    Args:
        message: Текст сообщения
        user_ids: Список ID пользователей (если None, то всем активным)
    """
    try:
        if user_ids is None:
            # Импортируем модули для работы с базой данных
            from database.models import SessionLocal
            from database import crud
            
            # Получаем список активных пользователей
            db = SessionLocal()
            active_users = db.query(crud.models.User).filter(crud.models.User.is_active == True).all()
            user_ids = [user.telegram_id for user in active_users]
            db.close()
        
        success_count = 0
        for user_id in user_ids:
            try:
                bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Ошибка при отправке сообщения пользователю {user_id}: {e}")
        
        logger.info(f"Массовая рассылка выполнена. Успешно отправлено {success_count} из {len(user_ids)} сообщений.")
        return success_count
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении массовой рассылки: {e}")
        return 0
