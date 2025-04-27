from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from config.db_config import DB_CONFIG

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

# Создание движка SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True)
    username = Column(String, index=True)
    first_name = Column(String)
    last_name = Column(String)
    risk_profile = Column(String)  # conservative, moderate, aggressive
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_active = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    portfolios = relationship("Portfolio", back_populates="user")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    weights = Column(JSON)  # {"TICKER": weight}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="portfolios")
    
class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    max_stocks = Column(Integer, default=10)
    max_bonds = Column(Integer, default=5)
    excluded_sectors = Column(JSON, default=list)  # ["sector1", "sector2"]
    preferred_sectors = Column(JSON, default=list)  # ["sector1", "sector2"]
    
    user = relationship("User", back_populates="preferences")

# Функция для создания всех таблиц
def create_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
