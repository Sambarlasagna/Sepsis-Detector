from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  

# Async engine
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Session factory
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, autoflush=False, autocommit=False
)

# Dependency for routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
