from datetime import datetime
from typing import Optional

import httpx
import re
from bson import ObjectId
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.models.database import get_db
from app.models.models import UserInDB


# ── Email + Password ──────────────────────────────────────────────────────────

async def signup_with_email(
    email:    str,
    password: str,
    name:     str,
) -> dict:
    """
    Create a new email/password user.
    Returns { access_token, user }.
    """
    db = get_db()

    # Duplicate check
    existing = await db.users.find_one({"email": email.lower()})
    if existing:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "An account with this email already exists",
        )

    # Enforce password strength
    if len(password) < 8 or len(password) > 16:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "Password must be 8-16 characters",
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "Password must contain at least one uppercase letter",
        )
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "Password must contain at least one lowercase letter",
        )
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\\|,.<>/?]", password):
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "Password must contain at least one special character",
        )

    now  = datetime.utcnow()
    user = {
        "email":         email.lower(),
        "name":          name.strip(),
        "password_hash": hash_password(password),
        "auth_provider": "email",
        "clerk_user_id": None,
        "avatar_url":    None,
        "is_active":     True,
        "is_verified":   False,   # set True after email verification
        "created_at":    now,
        "updated_at":    now,
        "last_login":    now,
    }

    result  = await db.users.insert_one(user)
    user_id = str(result.inserted_id)

    token = create_access_token(user_id=user_id, email=email)

    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":            user_id,
            "email":         email,
            "name":          name,
            "auth_provider": "email",
            "is_verified":   False,
        },
    }


async def login_with_email(email: str, password: str) -> dict:
    """
    Verify email + password, return JWT.
    Uses a timing-safe check to prevent user enumeration.
    """
    db   = get_db()
    user = await db.users.find_one({"email": email.lower()})

    # Always run verify_password even if user not found
    # (prevents timing-based user enumeration)
    dummy_hash = "$2b$12$dummyhashfortimingsafety000000000000000000000000000"
    stored_hash = user["password_hash"] if user else dummy_hash

    password_ok = verify_password(password, stored_hash)

    if not user or not password_ok:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid email or password",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Account is deactivated",
        )

    # Update last_login
    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}},
    )

    user_id = str(user["_id"])
    token   = create_access_token(user_id=user_id, email=email)

    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":            user_id,
            "email":         user["email"],
            "name":          user["name"],
            "auth_provider": user["auth_provider"],
            "avatar_url":    user.get("avatar_url"),
            "is_verified":   user.get("is_verified", False),
        },
    }


# ── Clerk / Google OAuth ──────────────────────────────────────────────────────

async def verify_clerk_token(
    clerk_session_token: str | None = None,
    clerk_user_id: str | None = None,
    email: str | None = None,
    name: str | None = None,
) -> dict:
    """
    Verify Clerk user and create/update user in our DB.
    Can either verify a session token OR accept user data directly from frontend.
    """
    # If we have direct user data from frontend, use it
    if clerk_user_id and email:
        print(f"Using direct user data: {clerk_user_id}, {email}")
        clerk_id = clerk_user_id
        user_email = email
        user_name = name or email.split("@")[0]
        avatar_url = None
        
        # Skip Clerk verification and go directly to DB
        return await _upsert_clerk_user(clerk_id, user_email, user_name, avatar_url)
    
    if not clerk_session_token:
        raise HTTPException(
            status_code = 400,
            detail      = "Either clerk_session_token or clerk_user_id+email is required",
        )

    if not settings.clerk_secret_key:
        raise HTTPException(
            status_code = 503,
            detail      = "Clerk is not configured on this server",
        )

    # Verify session token with Clerk
    try:
        async with httpx.AsyncClient() as client:
            verify_response = await client.post(
                f"https://api.clerk.com/v1/clerk/sessions/{clerk_session_token}/verify",
                headers={
                    "Authorization":  f"Bearer {settings.clerk_secret_key}",
                    "Content-Type":   "application/json",
                },
                json={},
                timeout=10,
            )
            print(f"Clerk verify response: {verify_response.status_code}")
            
            if verify_response.status_code != 200:
                raise HTTPException(
                    status_code = 401,
                    detail      = f"Invalid Clerk session token",
                )
            
            verify_data = verify_response.json()
            user_id = verify_data.get("user_id")
            
            user_response = await client.get(
                f"https://api.clerk.com/v1/users/{user_id}",
                headers={
                    "Authorization":  f"Bearer {settings.clerk_secret_key}",
                    "Content-Type":   "application/json",
                },
                timeout=10,
            )
            
            if user_response.status_code != 200:
                raise HTTPException(
                    status_code = 401,
                    detail      = "Failed to get user from Clerk",
                )
            
            clerk_user = user_response.json()
    except Exception as e:
        print(f"Clerk error: {e}")
        raise HTTPException(
            status_code = 502,
            detail      = f"Failed to verify with Clerk: {str(e)}",
        )

    clerk_id = clerk_user.get("id")
    email_data = clerk_user.get("email_addresses", [{}])[0]
    user_email = email_data.get("email_address", "") if email_data else ""
    user_name = (
        f"{clerk_user.get('first_name', '')} {clerk_user.get('last_name', '')}".strip()
        or user_email.split("@")[0]
    )
    avatar_url = clerk_user.get("image_url")

    if not user_email:
        raise HTTPException(
            status_code = 400,
            detail      = "Clerk user has no email address",
        )

    return await _upsert_clerk_user(clerk_id, user_email, user_name, avatar_url)


async def _upsert_clerk_user(clerk_id: str, email: str, name: str, avatar_url: str | None) -> dict:

    print("Attempting database connection...")
    try:
        db  = get_db()
        print("Database connected")
    except Exception as e:
        print(f"Database error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code = 500,
            detail      = f"Database connection failed: {str(e)}",
        )
    
    now = datetime.utcnow()

    # Upsert: update if clerk_user_id matches, insert if new
    result = await db.users.find_one_and_update(
        {"$or": [{"clerk_user_id": clerk_id}, {"email": email.lower()}]},
        {
            "$set": {
                "email":         email.lower(),
                "name":          name,
                "auth_provider": "clerk",
                "clerk_user_id": clerk_id,
                "avatar_url":    avatar_url,
                "is_active":     True,
                "is_verified":   True,
                "updated_at":    now,
                "last_login":    now,
            },
            "$setOnInsert": {
                "password_hash": None,
                "created_at":    now,
            },
        },
        upsert   = True,
        return_document = True,
    )

    user_id = str(result["_id"])
    token   = create_access_token(
        user_id = user_id,
        email   = email,
        extra   = {"provider": "clerk"},
    )

    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":            user_id,
            "email":         email,
            "name":          name,
            "auth_provider": "clerk",
            "avatar_url":    avatar_url,
            "is_verified":   True,
        },
    }