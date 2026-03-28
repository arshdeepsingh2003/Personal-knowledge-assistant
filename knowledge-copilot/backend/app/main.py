from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import ingest
from app.api import embed     
from app.api import ingest, embed, vectorstore,retriever,chat       

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(embed.router)
app.include_router(vectorstore.router)     
app.include_router(retriever.router)
app.include_router(chat.router) 

@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.app_version}