import asyncio
from sqlalchemy import text
from app.db.base import Base
from app.db.session import engine
import app.models

async def init_db() -> None:
    print("Initializing database...")
    async with engine.begin() as conn:
        print("Enabling vector extension...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        print("Creating database table and indexes...")
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())