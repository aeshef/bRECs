import logging
import sys
from datetime import datetime
from sqlalchemy.orm import Session

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/portfolio-advisor/logs/urgent_signals.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from database.models import SessionLocal
from database import crud
from telegram_bot.notification_sender import send_urgent_signal

def check_urgent_signals():
    """
    Проверяет наличие экстренных сигналов для портфелей пользователей
    """
    logger.info("Запуск проверки экстренных сигналов...")
    
    try:
        # Получаем актуальные данные о рынке
        market_data = fetch_latest_market_data()
        
        # Проверяем наличие важных новостей
        important_news = fetch_important_news()
        
        # Определяем, есть ли экстренные сигналы
        has_urgent_signals = detect_urgent_signals(market_data, important_news)
        
        if has_urgent_signals:
            logger.info("Обнаружены экстренные сигналы. Уведомляем пользователей...")
            
            # Получаем всех активных пользователей
            db = SessionLocal()
            active_users = db.query(crud.models.User).filter(crud.models.User.is_active == True).all()
            
            for user in active_users:
                try:
                    # Получаем текущий портфель пользователя
                    current_portfolio = crud.get_latest_portfolio(db, user.id)
                    
                    if current_portfolio:
                        # Генерируем рекомендации с учетом экстренных сигналов
                        urgent_recommendations = generate_urgent_recommendations(
                            current_portfolio.weights, 
                            market_data, 
                            important_news,
                            user.risk_profile
                        )
                        
                        # Обновляем портфель пользователя
                        crud.update_portfolio(db, current_portfolio.id, urgent_recommendations['weights'])
                        
                        # Отправляем уведомление пользователю
                        send_urgent_signal(user.telegram_id, urgent_recommendations)
                        
                        logger.info(f"Отправлен экстренный сигнал пользователю {user.telegram_id}")
                
                except Exception as e:
                    logger.error(f"Ошибка при обработке экстренного сигнала для пользователя {user.telegram_id}: {e}")
        
        else:
            logger.info("Экстренных сигналов не обнаружено.")
        
        logger.info("Проверка экстренных сигналов завершена успешно")
    
    except Exception as e:
        logger.error(f"Ошибка при проверке экстренных сигналов: {e}")
    
    finally:
        if 'db' in locals():
            db.close()

def fetch_latest_market_data():
    """
    Получает актуальные данные о рынке
    """
    try:
        # Импортируем ваши существующие модули для получения рыночных данных
        # Здесь нужно использовать ваш код для получения актуальных цен
        
        # Временная заглушка
        return {'market_volatility': 0.01, 'market_trend': 'stable'}
    
    except Exception as e:
        logger.error(f"Ошибка при получении рыночных данных: {e}")
        return {}

def fetch_important_news():
    """
    Получает важные новости, которые могут повлиять на рынок
    """
    try:
        # Импортируем ваши существующие модули для обработки новостей
        # from data_collection.news_processor.event_detector import detect_important_events
        
        # Временная заглушка
        return []
    
    except Exception as e:
        logger.error(f"Ошибка при получении важных новостей: {e}")
        return []

def detect_urgent_signals(market_data, important_news):
    """
    Определяет, есть ли экстренные сигналы на основе рыночных данных и новостей
    """
    # Пример логики определения экстренных сигналов
    high_volatility_threshold = 0.03  # 3% волатильность считается высокой
    
    # Проверяем волатильность рынка
    if market_data.get('market_volatility', 0) > high_volatility_threshold:
        return True
    
    # Проверяем наличие важных новостей с негативным сентиментом
    for news in important_news:
        if news.get('sentiment', 0) < -0.5:  # Сильно негативный сентимент
            return True
    
    return False

def generate_urgent_recommendations(current_weights, market_data, important_news, risk_profile):
    """
    Генерирует рекомендации с учетом экстренных сигналов
    """
    # Пример логики для генерации рекомендаций в случае экстренных ситуаций
    
    # Увеличиваем долю безрисковых активов в зависимости от риск-профиля
    safe_asset_increase = 0.2  # По умолчанию увеличиваем на 20%
    
    if risk_profile == "conservative":
        safe_asset_increase = 0.3  # Для консервативных увеличиваем на 30%
    elif risk_profile == "aggressive":
        safe_asset_increase = 0.1  # Для агрессивных увеличиваем только на 10%
    
    # Считаем текущую долю безрисковых активов
    current_safe_assets = sum([
        weight for ticker, weight in current_weights.items() 
        if is_safe_asset(ticker)
    ])
    
    # Новая целевая доля безрисковых активов
    target_safe_assets = min(0.9, current_safe_assets + safe_asset_increase)
    
    # Пересчитываем веса
    new_weights = {}
    
    # Сначала копируем текущие веса
    for ticker, weight in current_weights.items():
        new_weights[ticker] = weight
    
    # Корректируем веса в зависимости от типа актива
    scaling_factor = (1 - target_safe_assets) / (1 - current_safe_assets) if current_safe_assets < 1 else 0
    
    for ticker in new_weights:
        if is_safe_asset(ticker):
            # Увеличиваем долю безрисковых активов пропорционально
            if current_safe_assets > 0:
                new_weights[ticker] = new_weights[ticker] * (target_safe_assets / current_safe_assets)
        else:
            # Уменьшаем долю рисковых активов
            new_weights[ticker] = new_weights[ticker] * scaling_factor
    
    # Нормализуем веса
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        for ticker in new_weights:
            new_weights[ticker] = new_weights[ticker] / total_weight
    
    # Формируем текст рекомендации
    recommendation_text = "В связи с высокой рыночной волатильностью рекомендуется временно увеличить долю безрисковых активов."
    
    return {
        'weights': new_weights,
        'recommendation': recommendation_text,
        'reason': "Обнаружена высокая волатильность на рынке" if market_data.get('market_volatility', 0) > 0.03 else "Обнаружены важные новости, которые могут повлиять на рынок"
    }

def is_safe_asset(ticker):
    """
    Определяет, является ли актив безрисковым
    """
    # Список тикеров безрисковых активов (гос. облигации, фонды денежного рынка и т.д.)
    safe_assets = ["OFZ", "BOND", "CASH", "RUOB"]
    
    return ticker in safe_assets or "OFZ" in ticker or "BOND" in ticker

if __name__ == "__main__":
    check_urgent_signals()
