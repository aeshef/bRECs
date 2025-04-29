import os
import sys
import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import (create_engine, Column, Integer, String, Float,
                          DateTime, JSON, Boolean, ForeignKey, MetaData,
                          event, TypeDecorator)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.pool import QueuePool

try:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    PYS_PATH = PROJECT_ROOT / 'pys'
    if PYS_PATH.exists() and str(PYS_PATH) not in sys.path:
         sys.path.insert(0, str(PYS_PATH))
    from pys.utils.path_helper import get_project_root 
except ImportError:
     print("Warning: Could not import path_helper to setup sys.path in models.py")
     PROJECT_ROOT = Path(".")

env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

DB_ENGINE_TYPE = os.getenv('DB_ENGINE', 'postgresql')
DATABASE_URL = None

print(f"PROJECT_ROOT: {PROJECT_ROOT}") 
print(f"env_path: {env_path}")  
print(DB_ENGINE_TYPE)

if DB_ENGINE_TYPE == 'postgresql':
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Database credentials not fully configured in .env for PostgreSQL")
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DB_ENGINE_TYPE == 'sqlite':
    SQLITE_PATH = os.getenv('SQLITE_PATH', 'local_dev.db')
    sqlite_abs_path = PROJECT_ROOT / SQLITE_PATH
    DATABASE_URL = f"sqlite:///{sqlite_abs_path}"
    print(f"Using SQLite database at: {sqlite_abs_path}")
else:
    raise ValueError(f"Unsupported DB_ENGINE: {DB_ENGINE_TYPE}")

engine_args = {'echo': False}
if DB_ENGINE_TYPE == 'postgresql' and os.getenv('ENV_TYPE') == 'production':
    engine_args.update({
        'poolclass': QueuePool,
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 1800
    })

engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

if DB_ENGINE_TYPE == 'sqlite':
    class ForceJSON(TypeDecorator):
        impl = String
        cache_ok = True

        def process_bind_param(self, value, dialect):
            if value is not None:
                value = json.dumps(value)
            return value

        def process_result_value(self, value, dialect):
            if value is not None:
                 try:
                     value = json.loads(value)
                 except (json.JSONDecodeError):
                     pass
            return value
    JSON = ForceJSON

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, index=True, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    risk_profile = Column(String, index=True, nullable=True) # 'conservative', 'moderate', 'aggressive'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_active_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    # Связи
    portfolios = relationship("Portfolio", back_populates="user", order_by="desc(Portfolio.created_at)", cascade="all, delete-orphan")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, tg_id={self.telegram_id}, profile='{self.risk_profile}', active={self.is_active})>"

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=True)
    weights = Column(JSON, nullable=True) # Структура: {"TICKER": 0.15, ...}
    metrics = Column(JSON, nullable=True) # Структура: {"sharpe": 1.2, "return": 0.1, ...}
    strategy_profile = Column(String, nullable=True) # Профиль риска, использованный для этого расчета
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    # Связь
    user = relationship("User", back_populates="portfolios")

    def __repr__(self):
        return f"<Portfolio(id={self.id}, user_id={self.user_id}, date='{self.created_at.strftime('%Y-%m-%d')}')>"

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    max_stocks = Column(Integer, default=10)
    max_bonds = Column(Integer, default=5)
    excluded_sectors = Column(JSON, default=list)  # ["Нефть и газ", ...]
    preferred_sectors = Column(JSON, default=list)  # ["IT", "Финансы", ...]
    excluded_tickers = Column(JSON, default=list)  # ["SBER", "GAZP", ...]
    preferred_tickers = Column(JSON, default=list)  # ["YNDX", "PLZL", ...]
    allow_short = Column(Boolean, default=False)

    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreferences(id={self.id}, user_id={self.user_id})>"

def create_all_tables():
   print(f"Attempting to create tables for database engine: {DB_ENGINE_TYPE}")
   if DATABASE_URL:
       print(f"Database URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
   try:
       Base.metadata.create_all(bind=engine)
       print("Tables created successfully (if they didn't exist).")
   except Exception as e:
       print(f"!!! Error creating tables: {e}")
       print("!!! Please check database connection details in .env and ensure the database exists.")
       raise

if DB_ENGINE_TYPE == 'sqlite':
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON;")
        finally:
            cursor.close()
    print("SQLite PRAGMA foreign_keys=ON configured.")
