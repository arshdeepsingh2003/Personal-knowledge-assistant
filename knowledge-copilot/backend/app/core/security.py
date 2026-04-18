from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from app.core.config import settings


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(
    user_id:  str,
    email:    str,
    extra:    dict = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT.
    Payload: sub (user_id), email, iat, exp + any extra claims.
    """
    now     = datetime.utcnow()
    expires = now + (
        expires_delta
        or timedelta(minutes=settings.jwt_expire_minutes)
    )

    payload = {
        "sub":   str(user_id),
        "email": email,
        "iat":   now,
        "exp":   expires,
        **(extra or {}),
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT.
    Raises JWTError (caught by middleware) if invalid or expired.
    """
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )