import logging
import sys
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/opt/portfolio-advisor/logs/weekly_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к проекту в PYTHONPATH
sys.path.append('/opt/portfolio-advisor')

# Импортируем модули проекта
from database.models import SessionLocal
from database import crud
from telegram_bot.notification_sender import send_portfolio_update

def run_weekly_pipeline():
    """
    Запуск еженедельного пайплайна для обновления рекомендаций по портфелям
    """
    logger.info("Запуск еженедельного пайплайна...")
    
    try:
        # Получаем всех активных пользователей
        db = SessionLocal()
        active_users = db.query(crud.models.User).filter(crud.models.User.is_active == True).all()
        
        logger.info(f"Найдено {len(active_users)} активных пользователей для обновления портфелей")
        
        for user in active_users:
            try:
                # Получаем предпочтения пользователя
                preferences = crud.get_user_preferences(db, user.id)
                
                # Генерируем рекомендации для пользователя с учетом его предпочтений
                recommendations = generate_portfolio_recommendations(user, preferences)
                
                # Получаем текущий портфель пользователя
                current_portfolio = crud.get_latest_portfolio(db, user.id)
                
                # Обновляем или создаем портфель
                if current_portfolio:
                    updated_portfolio = crud.update_portfolio(db, current_portfolio.id, recommendations['weights'])
                else:
                    updated_portfolio = crud.create_portfolio(
                        db, 
                        user.id, 
                        f"Portfolio {datetime.now().strftime('%Y-%m-%d')}", 
                        recommendations['weights']
                    )
                
                # Отправляем уведомление пользователю
                if recommendations.get('significant_changes', False):
                    send_portfolio_update(user.telegram_id, recommendations, is_urgent=False)
                
                logger.info(f"Обновлен портфель для пользователя {user.telegram_id}")
            
            except Exception as e:
                logger.error(f"Ошибка при обработке пользователя {user.telegram_id}: {e}")
        
        logger.info("Еженедельный пайплайн завершен успешно")
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении еженедельного пайплайна: {e}")
    
    finally:
        db.close()

def generate_portfolio_recommendations(user, preferences):
    """
    Генерирует рекомендации по портфелю для пользователя на основе его предпочтений
    
    Это обертка над вашим существующим пайплайном. Здесь вы интегрируете ваш код.
    """
    try:
        # Импортируем ваши существующие модули для работы с данными и оптимизации
        from data_collection.market_data import run_pipeline_market
        from data_collection.market_cap import run_pipeline_market_cap
        from data_collection.fundamental_data import run_pipeline_fundamental
        from data_collection.tech_analysis import run_pipeline_technical
        from data_collection.data_integration import run_pipeline_integration
        from porfolio_optimization.signal_generator import run_pipeline_signal_generator
        from porfolio_optimization.portfolio_optimizer import PortfolioOptimizer
        
        # Определяем параметры для запуска пайплайна
        risk_free_rate = 0.075  # TODO: получать актуальную безрисковую ставку
        
        # Определяем даты для анализа
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # данные за год
        
        # Адаптируем параметры на основе риск-профиля пользователя
        weight_tech = 0.5
        weight_sentiment = 0.3
        weight_fundamental = 0.2
        min_rf_allocation = 0.3
        max_rf_allocation = 0.5
        
        if user.risk_profile == "conservative":
            min_rf_allocation = 0.5
            max_rf_allocation = 0.7
            weight_fundamental = 0.4
            weight_tech = 0.4
            weight_sentiment = 0.2
        elif user.risk_profile == "aggressive":
            min_rf_allocation = 0.1
            max_rf_allocation = 0.3
            weight_tech = 0.6
            weight_sentiment = 0.3
            weight_fundamental = 0.1
        
        # Формируем список тикеров с учетом предпочтений пользователя
        tickers_list = get_tickers_based_on_preferences(preferences)
        
        # Запуск пайплайна
        run_pipeline_market(
            tickers=tickers_list,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            token="YOUR_TOKEN"  # Замените на ваш токен
        )
        
        market_caps_df = run_pipeline_market_cap(
            base_path="/opt/portfolio-advisor/data",
            tickers=tickers_list
        )
        
        run_pipeline_fundamental(ticker_list=tickers_list)
        run_pipeline_technical(tickers=tickers_list)
        
        # Здесь должен быть вызов для новостного пайплайна
        # Так как он использует monkey patching, нужно адаптировать его
        
        run_pipeline_integration(tickers=tickers_list)
        
        # Генерация сигналов с настроенными весами
        run_pipeline_signal_generator(
            weight_tech=weight_tech,
            weight_sentiment=weight_sentiment,
            weight_fundamental=weight_fundamental
        )
        
        # Оптимизация портфеля
        portfolio = PortfolioOptimizer(
            risk_free_rate=risk_free_rate,
            min_rf_allocation=min_rf_allocation,
            max_rf_allocation=max_rf_allocation
        ).run_pipeline()
        
        # Определяем, есть ли значительные изменения по сравнению с текущим портфелем
        significant_changes = check_significant_changes(portfolio['weights'], user.id)
        
        # Возвращаем рекомендации
        return {
            'weights': portfolio['weights'],
            'metrics': portfolio.get('metrics', {}),
            'significant_changes': significant_changes
        }
    
    except Exception as e:
        logger.error(f"Ошибка при генерации рекомендаций: {e}")
        # Возвращаем пустые рекомендации в случае ошибки
        return {
            'weights': {},
            'metrics': {},
            'significant_changes': False,
            'error': str(e)
        }

def get_tickers_based_on_preferences(preferences):
    """
    Формирует список тикеров с учетом предпочтений пользователя
    """
    # Базовый список тикеров (например, топ-100 российских акций)
    base_tickers = [
        "SBER", "GAZP", "LKOH", "GMKN", "ROSN", 
        "TATN", "MGNT", "NVTK", "PLZL", "POLY",
        # Добавьте больше тикеров по умолчанию
    ]
    
    if not preferences:
        return base_tickers
    
    # Исключаем нежелательные секторы
    if preferences.excluded_sectors:
        # Логика исключения секторов
        pass
    
    # Добавляем приоритетные секторы
    if preferences.preferred_sectors:
        # Логика добавления приоритетных секторов
        pass
    
    # Ограничиваем количество акций
    max_stocks = preferences.max_stocks or 10
    if len(base_tickers) > max_stocks:
        return base_tickers[:max_stocks]
    
    return base_tickers

def check_significant_changes(new_weights, user_id):
    """
    Проверяет, есть ли значительные изменения по сравнению с текущим портфелем
    """
    threshold = 0.05  # 5% изменение считается значительным
    
    try:
        db = SessionLocal()
        current_portfolio = crud.get_latest_portfolio(db, user_id)
        
        if not current_portfolio:
            return True  # Если нет текущего портфеля, считаем изменения значительными
        
        current_weights = current_portfolio.weights
        
        # Объединяем все тикеры из обоих портфелей
        all_tickers = set(current_weights.keys()) | set(new_weights.keys())
        
        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0)
            new_weight = new_weights.get(ticker, 0)
            
            if abs(current_weight - new_weight) > threshold:
                return True  # Найдено значительное изменение
        
        return False  # Нет значительных изменений
    
    except Exception as e:
        logger.error(f"Ошибка при проверке значительных изменений: {e}")
        return True  # В случае ошибки считаем изменения значительными
    
    finally:
        db.close()

if __name__ == "__main__":
    run_weekly_pipeline()
