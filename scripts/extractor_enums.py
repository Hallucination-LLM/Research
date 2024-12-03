from enum import Enum

class DfColumnsEnum(str, Enum):
    IDX = 'index'
    PROBLEMATIC = 'problematic_spans'
    RESPONSE = 'model_response'
    HALLU_IDX = 'hallu_indices'
    HALLU_TOKENS = 'hallu_tokens'
    CONTAIN_HALLU = 'contain_hallu'
    CONTEXT = 'formatted_context'
    GPT_INDEX = 'gpt_index'
    PROMPT_LEN = 'prompt_length'

    def __str__(self) -> str:
        return str.__str__(self)

class EvalDfColumnsEnum(str, Enum):
    PROBLEMATIC = 'problematic_spans'
    RESPONSE = 'response'

    def __str__(self) -> str:
        return str.__str__(self)
