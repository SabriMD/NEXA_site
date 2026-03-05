from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Servir les screenshots et assets
if os.path.exists("screenshots"):
    app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}
