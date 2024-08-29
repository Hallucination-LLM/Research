import torch
import logging
import os

from fastapi import Request
from src.detector import load_detector
from src.llm_module import load_model, prepare_prompt, get_hidden_states, get_local_llms_response, get_text_streamer
from app.fastapi_models import DetectHalluRequest, DetectHalluResponse, RequestModel, ResponseModel
from config import *

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

DETECTOR_WEIGHTS_PATH = os.path.join(MODELS_DIR, HALLU_DETECTOR_FILE)


class AIProcessingService:
    def __init__(self):
        self.device = None
        self.detector = None
        self.llm = None
        self.tokenizer = None

    async def on_startup(self):
        self._set_device()
        await self._load_llm()
        await self._load_detector()

    async def on_shutdown(self):
        pass

    def _set_device(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")

    async def _load_llm(self):
        self.llm, self.tokenizer = load_model(MODEL_ID, self.device, USE_UNSLOTH)
        logger.info(f"LLM model loaded: {MODEL_ID}")

    async def _load_detector(self):
        self.detector = load_detector(DETECTOR_WEIGHTS_PATH, self.device)
        logger.info(f"Hallucination detector loaded: {DETECTOR_WEIGHTS_PATH}")


    async def detect_hallu(self, request: DetectHalluRequest):

        logger.info(f'{self.llm.dtype = }')

        prompt = prepare_prompt(self.tokenizer, request.user_input, request.system_msg)
        logger.info(f"{prompt = }")

        hidden_states = get_hidden_states(
            model=self.llm,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device
        )

        is_hallucination = self.detector.predict(hidden_states.to(torch.float32))
        logger.info(f"{is_hallucination = }")
        return DetectHalluResponse(user_input=request.user_input, hallucination=is_hallucination)
    
    async def generate(self, request: RequestModel):
            
        prompt = prepare_prompt(self.tokenizer, request.user_input, "")
        logger.info(f"{prompt = }")

        text_streamer = get_text_streamer(
            tokenizer=self.tokenizer,
            skip_prompt=True
        )

        text_streamer = get_local_llms_response(
            model=self.llm,
            tokenizer=self.tokenizer,
            text_streamer=text_streamer,
            prompt=prompt,
            max_new_tokens=2048,
            temperature=0.0,
            device=self.device,
            dola=request.use_dola
        )
        
        generated_text = "".join([text for text in text_streamer])
        return ResponseModel(message=f"Generated content: {generated_text}")
    
async def get_ai_processing_service(request: Request) -> AIProcessingService:
    return request.app.state.ai_processing_service