from app.core.config import settings 
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Async Session
engine = create_async_engine(
    settings.async_database_url,
    echo=(settings.ENVIRONMENT=="development"),
    pool_pre_ping=True
)

AsyncSessionLocal = async_sessionmaker(
    bind = engine,
    class_= AsyncSession,
    expire_on_commit = False,
    autoflush=False,
)

# Sync Session
sync_engine = create_engine(
    settings.sync_database_url,
    pool_pre_ping=True
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()