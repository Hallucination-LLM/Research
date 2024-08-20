from pydantic import BaseModel
from typing import Optional
from config import SYSTEM_MSG

# Sample request model
class RequestModel(BaseModel):
    input: str
    use_dola: Optional[bool] = False

# Sample response model
class ResponseModel(BaseModel):
    message: str

class DetectHalluRequest(BaseModel):
    user_input: str
    system_msg: Optional[str] = SYSTEM_MSG

class DetectHalluResponse(BaseModel):
    input: str
    hallucination: int