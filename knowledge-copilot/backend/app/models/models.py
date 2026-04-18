from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId


# ObjectId helper 
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError(f"Invalid ObjectId: {v}")
        return str(v)


# User 
class UserInDB(BaseModel):
    id:            Optional[PyObjectId] = Field(None, alias="_id")
    email:         EmailStr
    name:          str
    password_hash: Optional[str]        = None
    auth_provider: Literal["email", "google", "clerk"] = "email"
    clerk_user_id: Optional[str]        = None
    avatar_url:    Optional[str]        = None
    is_active:     bool                 = True
    is_verified:   bool                 = False
    created_at:    datetime             = Field(default_factory=datetime.utcnow)
    updated_at:    datetime             = Field(default_factory=datetime.utcnow)
    last_login:    Optional[datetime]   = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserPublic(BaseModel):
    """Safe user object returned to the frontend — no password_hash."""
    id:            str
    email:         EmailStr
    name:          str
    auth_provider: str
    avatar_url:    Optional[str] = None
    is_verified:   bool
    created_at:    datetime


# Chat session 
class ChatSessionInDB(BaseModel):
    id:         Optional[PyObjectId] = Field(None, alias="_id")
    user_id:    str
    title:      str = "New conversation"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active:  bool     = True

    class Config:
        populate_by_name     = True
        arbitrary_types_allowed = True
        json_encoders        = {ObjectId: str}


# Chat message 
class SourceRef(BaseModel):
    file_name: str
    page:      Optional[int]  = None
    score:     float          = 0.0


class ChatMessageInDB(BaseModel):
    id:         Optional[PyObjectId] = Field(None, alias="_id")
    session_id: str
    user_id:    str
    role:       Literal["user", "assistant"]
    content:    str
    sources:    list[SourceRef] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name     = True
        arbitrary_types_allowed = True
        json_encoders        = {ObjectId: str}