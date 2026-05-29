import logging
import traceback

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("knowledge_copilot.errors")

# It ensures that every error in your backend returns in the same format
# Every error response in the app has this exact shape:
# {
#   "error": {
#     "code":    "VALIDATION_ERROR",
#     "message": "chunk_size must be between 100 and 8000",
#     "detail":  { ... }          ← optional extra info
#   }
# }

def error_response(
    code:    str,
    message: str,
    status_code: int  = 400,
    detail:  dict     = None,
) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


#Global exception handlers (registered on the FastAPI app)

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Replaces FastAPI's default 422 with our consistent format."""
    return error_response(
        code        = "VALIDATION_ERROR",
        message     = "Request validation failed",
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail      = {"errors": exc.errors()},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — logs the full traceback."""
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path,
        exc, traceback.format_exc(),
    )
    return error_response(
        code        = "INTERNAL_ERROR",
        message     = str(exc) if str(exc) else "An unexpected error occurred",
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail      = {"type": type(exc).__name__, "detail": str(exc)},
    )