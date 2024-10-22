from pydantic import BaseModel
from enum import Enum
from typing import List
from pydantic.fields import Field

class DataType(str, Enum):
    SUMMARIZATION = 'summarization'
    QA = 'q_and_a'

    def __str__(self):
        return self.value

class HallucinationAnswer(str, Enum):
    YES = "YES"
    NO = "NO"

class HallucinationSchema(BaseModel):
    answer: HallucinationAnswer

class ProblematicSpan(BaseModel):
    span: str

class EvaluationResult(BaseModel):
    decision: HallucinationAnswer
    explanation: str
    problematic_spans: List[ProblematicSpan] = Field(default_factory=list)