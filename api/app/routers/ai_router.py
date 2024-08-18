
from fastapi import APIRouter, Depends
from app.fastapi_models import DetectHalluRequest, DetectHalluResponse, RequestModel, ResponseModel
from app.services.ai_processing_service import get_ai_processing_service, AIProcessingService

router = APIRouter(tags=["ai"])


@router.post("/detect_hallu", response_model=DetectHalluResponse)
async def detect_hallu(request: DetectHalluRequest, ai_service: AIProcessingService = Depends(get_ai_processing_service)):
    return await ai_service.detect_hallu(request)


@router.post("/generate")
async def generate(request: RequestModel, ai_service: AIProcessingService = Depends(get_ai_processing_service)):
    return await ai_service.generate(request)