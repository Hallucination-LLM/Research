from pydantic import BaseModel
from enum import Enum
from typing import List
from pydantic.fields import Field

class DataType(str, Enum):
    SUMMARIZATION = 'summarization'
    QA = 'q_and_a'

    def __str__(self):
        return self.value

class IsAnswerCorrect(str, Enum):
    YES = "YES"
    NO = "NO"

class HallucinationSchema(BaseModel):
    answer: IsAnswerCorrect

class ProblematicSpan(BaseModel):
    span: str

class EvaluationResult(BaseModel):
    decision: IsAnswerCorrect
    explanation: str
    problematic_spans: List[ProblematicSpan] = Field(default_factory=list)

class OutputData(BaseModel):
    index: int
    document: str
    ground_truth: str
    response: str
    decision: IsAnswerCorrect
    gpt4_explanation: str
    problematic_spans: List[str]
    cost: float
    prompt: str