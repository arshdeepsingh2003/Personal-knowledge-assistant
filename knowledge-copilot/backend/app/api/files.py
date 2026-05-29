from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth_middleware import get_current_user
from app.services.document_loader import (
    delete_uploaded_file,
    get_signed_download_url,
    list_uploaded_files,
)

router = APIRouter(prefix="/api/v1/files", tags=["files"])


@router.get("")
async def list_files(current_user: dict = Depends(get_current_user)):
    files = await list_uploaded_files(user_id=current_user["id"])
    return {"files": files}


@router.get("/{file_id}/download")
async def download_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    url = await get_signed_download_url(file_id, user_id=current_user["id"])
    if not url:
        raise HTTPException(status_code=404, detail="File not found")
    return {"signed_url": url}


@router.delete("/{file_id}")
async def remove_file(
    file_id: str,
    current_user: dict = Depends(get_current_user),
):
    ok = await delete_uploaded_file(file_id, user_id=current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File deleted"}
