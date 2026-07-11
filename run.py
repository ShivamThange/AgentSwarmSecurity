import uvicorn

from twin.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app:app", host=settings.host, port=settings.port,
                reload=False)
