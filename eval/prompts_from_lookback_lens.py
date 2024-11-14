from enum import Enum
from evaluation_schema import DataType

DATA_RESPONSE_NAMES = {
    DataType.SUMMARIZATION: 'Summary',
    DataType.QA: 'Answer'
}

DATA_RESPONSE_NAMES_GT = {
    DataType.SUMMARIZATION: 'Summary',
    DataType.QA: 'Answers (a list of valid answers)'
}

EVAL_PROMPT_BEFORE = {
    DataType.SUMMARIZATION: """
    You will be provided with a document and a proposed summary. 
    Your task is to determine if the proposed summary can be directly inferred from the document. 
    If the summary contains any information not found in the document, it is considered hallucination. 
    Even if the summary is different from a ground truth summary, it might still be accurate, as long as it doesn't contain hallucinated information.\n
    For each proposed summary, explain why it contains hallucination or not based on the information from the document. 
    Focus only on the original document's content, disregarding any external context.
    """,
    DataType.QA: """
    You will be provided with a document and a proposed answer to a question. 
    Your task is to determine if the proposed answer can be directly inferred from the document. 
    If the answer contains any information not found in the document, it is considered hallucination. 
    Even if the answer is different from a ground truth answer, it might still be accurate, as long as it doesn't contain hallucinated information.\n
    For each proposed answer, explain why it contains hallucination or not based on the information from the document. 
    Focus only on the original document's content, disregarding any external context.
    """
}

EVAL_PROMPT_AFTER = {
    DataType.SUMMARIZATION: "Write your explanation first, and then give your final conclusion.",
    DataType.QA: "Write your explanation first, and then give your final conclusion."
}