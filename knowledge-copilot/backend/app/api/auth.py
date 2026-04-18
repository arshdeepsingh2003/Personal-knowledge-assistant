from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, field_validator
import re

from app.middleware.auth_middleware import get_current_user
from app.services.auth_service import (
    login_with_email,
    signup_with_email,
    verify_clerk_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request schemas ───────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v):
        if not v.strip():
            raise ValueError("Name must not be blank")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8 or len(v) > 16:
            raise ValueError("Password must be 8-16 characters")

        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")

        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")

        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\\|,.<>/?]", v):
            raise ValueError("Password must contain at least one special character")

        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ClerkTokenRequest(BaseModel):
    clerk_session_token: str | None = None
    clerk_user_id: str | None = None
    email: str | None = None
    name: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/signup", status_code=201)
async def signup(body: SignupRequest):
    """
    Register a new user with email + password.
    Returns JWT access token immediately — no separate login step needed.
    """
    return await signup_with_email(
        email=str(body.email),
        password=body.password,
        name=body.name,
    )


@router.post("/login")
async def login(body: LoginRequest):
    """
    Authenticate with email + password.
    Returns JWT access token.
    """
    return await login_with_email(
        email=str(body.email),
        password=body.password,
    )


@router.post("/clerk")
async def clerk_auth(body: ClerkTokenRequest):
    """
    Exchange Clerk session token OR user data for our own JWT.
    Frontend calls this after Clerk.signIn() completes.
    """
    return await verify_clerk_token(
        clerk_session_token=body.clerk_session_token,
        clerk_user_id=body.clerk_user_id,
        email=body.email,
        name=body.name,
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the currently authenticated user's profile.
    Requires: Authorization: Bearer <token>
    """
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "auth_provider": current_user.get("auth_provider"),
        "avatar_url": current_user.get("avatar_url"),
        "is_verified": current_user.get("is_verified", False),
        "created_at": current_user.get("created_at"),
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Stateless JWT logout — frontend deletes token.
    This endpoint exists so the frontend has a consistent API surface.
    For server-side invalidation, add token to a Redis blocklist here.
    """
    return {"message": "Logged out successfully"}