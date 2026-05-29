import uuid
from typing import Optional

from supabase import Client, create_client

from app.core.config import settings

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def generate_storage_path(filename: str, user_id: Optional[str] = None) -> str:
    prefix = f"users/{user_id}" if user_id else "uploads"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    unique_id = uuid.uuid4().hex
    safe_name = filename.rsplit(".", 1)[0].replace(" ", "_")[:60]
    return f"{prefix}/{unique_id}_{safe_name}.{ext}" if ext else f"{prefix}/{unique_id}_{safe_name}"


def upload_file(file_bytes: bytes, storage_path: str, content_type: Optional[str] = None) -> str:
    client = get_client()
    opts = {}
    if content_type:
        opts["content-type"] = content_type
    client.storage.from_(settings.supabase_bucket).upload(
        path=storage_path,
        file=file_bytes,
        file_options=opts,
    )
    return storage_path


def download_file(storage_path: str) -> bytes:
    client = get_client()
    return client.storage.from_(settings.supabase_bucket).download(storage_path)


def delete_file(storage_path: str) -> None:
    client = get_client()
    client.storage.from_(settings.supabase_bucket).remove([storage_path])


def get_signed_url(storage_path: str, expiry_seconds: int = 3600) -> str:
    client = get_client()
    result = client.storage.from_(settings.supabase_bucket).create_signed_url(
        path=storage_path,
        expires_in=expiry_seconds,
    )
    return result["signedURL"]
