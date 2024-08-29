from pydantic import BaseModel
from typing import Optional
from config import SYSTEM_MSG, USER_INPUT

# Sample request model
class RequestModel(BaseModel):
    user_input: Optional[str] = USER_INPUT
    system_msg: Optional[str] = SYSTEM_MSG
    use_dola: Optional[bool] = False

# Sample response model
class ResponseModel(BaseModel):
    message: str

class DetectHalluRequest(BaseModel):
    user_input: Optional[str] = USER_INPUT
    system_msg: Optional[str] = SYSTEM_MSG

class DetectHalluResponse(BaseModel):
    user_input: str
    hallucination: int