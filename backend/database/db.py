from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/autolens.db")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    actual_class = Column(String, nullable=False)
    correct = Column(Boolean, nullable=False)
    mode = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
