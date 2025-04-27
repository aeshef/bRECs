from sqlalchemy.orm import Session
from . import models
import datetime

# Функции для работы с пользователями
def get_user_by_telegram_id(db: Session, telegram_id: int):
    return db.query(models.User).filter(models.User.telegram_id == telegram_id).first()

def create_user(db: Session, telegram_id: int, username: str, first_name: str, last_name: str):
    db_user = models.User(
        telegram_id=telegram_id,
        username=username,
        first_name=first_name,
        last_name=last_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_risk_profile(db: Session, telegram_id: int, risk_profile: str):
    db_user = get_user_by_telegram_id(db, telegram_id)
    if db_user:
        db_user.risk_profile = risk_profile
        db_user.last_active = datetime.datetime.utcnow()
        db.commit()
        db.refresh(db_user)
    return db_user

# Функции для работы с портфелями
def get_latest_portfolio(db: Session, user_id: int):
    return db.query(models.Portfolio).filter(
        models.Portfolio.user_id == user_id,
        models.Portfolio.is_active == True
    ).order_by(models.Portfolio.updated_at.desc()).first()

def create_portfolio(db: Session, user_id: int, name: str, weights: dict):
    db_portfolio = models.Portfolio(
        user_id=user_id,
        name=name,
        weights=weights
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def update_portfolio(db: Session, portfolio_id: int, weights: dict):
    db_portfolio = db.query(models.Portfolio).filter(models.Portfolio.id == portfolio_id).first()
    if db_portfolio:
        db_portfolio.weights = weights
        db_portfolio.updated_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(db_portfolio)
    return db_portfolio

# Функции для работы с предпочтениями пользователей
def get_user_preferences(db: Session, user_id: int):
    return db.query(models.UserPreferences).filter(models.UserPreferences.user_id == user_id).first()

def create_or_update_preferences(db: Session, user_id: int, max_stocks: int = None, 
                               max_bonds: int = None, excluded_sectors: list = None, 
                               preferred_sectors: list = None):
    db_prefs = get_user_preferences(db, user_id)
    
    if not db_prefs:
        db_prefs = models.UserPreferences(user_id=user_id)
        db.add(db_prefs)
    
    if max_stocks is not None:
        db_prefs.max_stocks = max_stocks
    if max_bonds is not None:
        db_prefs.max_bonds = max_bonds
    if excluded_sectors is not None:
        db_prefs.excluded_sectors = excluded_sectors
    if preferred_sectors is not None:
        db_prefs.preferred_sectors = preferred_sectors
    
    db.commit()
    db.refresh(db_prefs)
    return db_prefs
