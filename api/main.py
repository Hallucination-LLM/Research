from fastapi import FastAPI
import os
from src.init_logger import init_logger
from contextlib import asynccontextmanager
from config import LOGGER_LEVEL, LOGGER_FORMAT, TOKENIZERS_PARALLELISM
from app.services.ai_processing_service import AIProcessingService
from app.routers import test_router, ai_router

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM

# Initialize logger
logger = init_logger(LOGGER_LEVEL, LOGGER_FORMAT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ai_processing_service = AIProcessingService()
    await app.state.ai_processing_service.on_startup()

    yield

    # Clean up
    await app.state.ai_processing_service.on_shutdown()


app = FastAPI(lifespan=lifespan)

app.include_router(test_router)
app.include_router(ai_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
