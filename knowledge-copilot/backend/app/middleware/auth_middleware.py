from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from app.core.security import decode_access_token
from app.models.database import get_db

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    FastAPI dependency — extracts and validates the Bearer JWT.

    Usage:
        @router.get("/protected")
        async def handler(user = Depends(get_current_user)):
            return {"user_id": user["id"]}

    Raises 401 if token is missing, malformed, or expired.
    """
    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Invalid or expired token",
        headers     = {"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Fetch live user record (detects deactivated accounts)
    db   = get_db()
    from bson import ObjectId
    user = await db.users.find_one({"_id": ObjectId(user_id)})

    if user is None:
        raise credentials_exception

    if not user.get("is_active", True):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Account is deactivated",
        )

    # Normalise _id → id string for downstream use
    user["id"] = str(user["_id"])
    return user


# Alias for optional auth (returns None instead of raising)
async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        HTTPBearer(auto_error=False)
    ),
):
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None