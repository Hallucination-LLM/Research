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
    DataType.SUMMARIZATION: (
        "You will be provided with a document and a proposed summary. Your task is to determine if the proposed summary can be directly inferred from the document. "
        "If the summary contains any information not found in the document, it is considered false. Even if the summary is different from a ground truth summary, "
        "it might still be true, as long as it doesn't contain false information.\nFor each proposed summary, explain why it is true or false based on the information from the document. "
        "Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed summary "
        "is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact "
        "phrases or name entities from the summary that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the summary, in Python list of strings format].**"
    ),
    DataType.QA: (
        "You will be provided with a document and a proposed answer to a question. Your task is to determine if the proposed answer can be directly inferred from the document. "
        "If the answer contains any information not found in the document, it is considered false. Even if the answer is different from a ground truth answer, it might still be true, "
        "as long as it doesn't contain false information.\nFor each proposed answer, explain why it is true or false based on the information from the document. "
        "Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed answer "
        "is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact "
        "phrases or name entities from the answer that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the answer, in Python list of strings format].**"
    )
}

EVAL_PROMPT_AFTER = {
    DataType.SUMMARIZATION: (
        "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed summary is completely accurate based on the document, "
        "or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the summary, in a list of strings]** if your conclusion is 'False'."
    ),
    DataType.QA: (
        "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, "
        "or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the answer, in a list of strings]** if your conclusion is 'False'."
    )
}
