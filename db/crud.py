import datetime
from sqlalchemy.orm import Session
from . import models
from typing import Optional

def get_user_by_telegram_id(db: Session, telegram_id: int) -> Optional[models.User]:
    """Находит пользователя по его Telegram ID."""
    return db.query(models.User).filter(models.User.telegram_id == telegram_id).first()

def create_user(db: Session, telegram_id: int, username: Optional[str] = None, first_name: Optional[str] = None, last_name: Optional[str] = None) -> Optional[models.User]:
    """Создает нового пользователя и его пустые предпочтения."""
    db_user = models.User(
        telegram_id=telegram_id,
        username=username,
        first_name=first_name,
        last_name=last_name,
        is_active=True, # Активируем при создании
        last_active_at=datetime.datetime.utcnow()
    )
    db.add(db_user)
    db_prefs = models.UserPreferences(
        user=db_user,
        allow_short=False
    )
    db.add(db_prefs)
    try:
        db.commit()
        db.refresh(db_user)
        db.refresh(db_prefs)
        return db_user
    except Exception as e:
        db.rollback() # Откатываем транзакцию в случае ошибки
        print(f"Error creating user: {e}") # Логирование ошибки
        raise # Передаем исключение дальше

def update_user_risk_profile(db: Session, telegram_id: int, risk_profile: str) -> Optional[models.User]:
    """Обновляет риск-профиль пользователя."""
    db_user = get_user_by_telegram_id(db, telegram_id)
    if db_user:
        db_user.risk_profile = risk_profile
        db_user.last_active_at = datetime.datetime.utcnow()
        try:
            db.commit()
            db.refresh(db_user)
            return db_user
        except Exception as e:
            db.rollback()
            print(f"Error updating risk profile for user {telegram_id}: {e}")
            raise
    return None

def update_user_active_status(db: Session, telegram_id: int, is_active: bool) -> Optional[models.User]:
    """Активирует или деактивирует пользователя."""
    db_user = get_user_by_telegram_id(db, telegram_id)
    if db_user:
        db_user.is_active = is_active
        db_user.last_active_at = datetime.datetime.utcnow()
        try:
            db.commit()
            db.refresh(db_user)
            return db_user
        except Exception as e:
            db.rollback()
            print(f"Error updating active status for user {telegram_id}: {e}")
            raise
    return None

def get_active_users(db: Session) -> Optional[list[models.User]]:
    """Возвращает список всех активных пользователей."""
    return db.query(models.User).filter(models.User.is_active == True).all()

def get_latest_portfolio(db: Session, user_id: int) -> Optional[models.Portfolio]:
    """Возвращает самый последний созданный портфель для пользователя."""
    return db.query(models.Portfolio)\
             .filter(models.Portfolio.user_id == user_id)\
             .order_by(models.Portfolio.created_at.desc())\
             .first()

def create_portfolio(db: Session, user_id: int, name: Optional[str] = None, weights: Optional[dict] = None, metrics: Optional[dict] = None, strategy_profile: Optional[str] = None) -> Optional[models.Portfolio]:
    """Создает новую запись портфеля в базе данных."""
    db_portfolio = models.Portfolio(
        user_id=user_id,
        name=name,
        weights=weights,
        metrics=metrics,
        strategy_profile=strategy_profile,
        created_at=datetime.datetime.utcnow()
    )
    db.add(db_portfolio)
    try:
        db.commit()
        db.refresh(db_portfolio)
        return db_portfolio
    except Exception as e:
        db.rollback()
        print(f"Error creating portfolio for user {user_id}: {e}")
        raise

def get_user_preferences(db: Session, user_id: int) -> Optional[models.UserPreferences]:
    """Получает предпочтения пользователя."""
    return db.query(models.UserPreferences).filter(models.UserPreferences.user_id == user_id).first()

def create_or_update_preferences(
    db: Session, 
    user_id: int, 
    max_stocks: Optional[int] = None, 
    max_bonds: Optional[int] = None, 
    excluded_sectors: Optional[list] = None, 
    preferred_sectors: Optional[list] = None,
    excluded_tickers: Optional[list] = None,
    preferred_tickers: Optional[list] = None,
    allow_short: Optional[bool] = None
) -> Optional[models.UserPreferences]:
    """Создает или обновляет предпочтения пользователя."""
    db_prefs = get_user_preferences(db, user_id)

    if not db_prefs:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            print(f"Cannot create/update preferences: User {user_id} not found.")
            return None
        print(f"Warning: Creating missing preferences for user {user_id}")
        db_prefs = models.UserPreferences(user_id=user_id)
        db.add(db_prefs)
        if max_stocks is None: 
            max_stocks = models.UserPreferences.max_stocks.default.arg
        if max_bonds is None: 
            max_bonds = models.UserPreferences.max_bonds.default.arg
        if excluded_sectors is None: 
            excluded_sectors = []
        if preferred_sectors is None: 
            preferred_sectors = []
        if excluded_tickers is None:
            excluded_tickers = []
        if preferred_tickers is None:
            preferred_tickers = []
        if allow_short is None:
            allow_short = models.UserPreferences.allow_short.default.arg

    if max_stocks is not None:
        db_prefs.max_stocks = max_stocks
    if max_bonds is not None:
        db_prefs.max_bonds = max_bonds
    if excluded_sectors is not None:
        db_prefs.excluded_sectors = [str(s) for s in excluded_sectors if isinstance(s, (str, int, float))]
    if preferred_sectors is not None:
        db_prefs.preferred_sectors = [str(s) for s in preferred_sectors if isinstance(s, (str, int, float))]
    if excluded_tickers is not None:
        db_prefs.excluded_tickers = [str(t) for t in excluded_tickers if isinstance(t, (str, int, float))]
    if preferred_tickers is not None:
        db_prefs.preferred_tickers = [str(t) for t in preferred_tickers if isinstance(t, (str, int, float))]
    if allow_short is not None:
        db_prefs.allow_short = allow_short

    try:
        db.commit()
        db.refresh(db_prefs)
        return db_prefs
    except Exception as e:
        db.rollback()
        print(f"Error creating/updating preferences for user {user_id}: {e}")
        raise
