from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_url)
    return _client


def get_db():
    return get_client()[settings.mongodb_db_name]


async def create_indexes():
    """
    Create all required MongoDB indexes on startup.
    Safe to call repeatedly — MongoDB ignores duplicates.
    """
    db = get_db()

    # users
    await db.users.create_index("email", unique=True)
    await db.users.create_index(
        "clerk_user_id", unique=True, sparse=True
    )

    # chat_sessions
    await db.chat_sessions.create_index(
        [("user_id", 1), ("created_at", -1)]
    )

    # chat_messages
    await db.chat_messages.create_index(
        [("session_id", 1), ("created_at", 1)]
    )
    await db.chat_messages.create_index("user_id")

    print("✓ MongoDB indexes created")