from .models import Base, engine, SessionLocal, User, Portfolio, UserPreferences
from .crud import (
    get_user_by_telegram_id,
    create_user,
    update_user_risk_profile,
    update_user_active_status,
    get_active_users,
    get_latest_portfolio,
    create_portfolio,
    get_user_preferences,
    create_or_update_preferences
)

from contextlib import contextmanager
from sqlalchemy.orm import Session

@contextmanager
def get_db_session() -> Session:
    """Контекстный менеджер для получения сессии SQLAlchemy."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
