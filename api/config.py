LOGGER_LEVEL = "INFO"
LOGGER_FORMAT = '[%(asctime)s] - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s'
TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'

DATA_DIR = 'data'
MODELS_DIR = 'models'
METRICS_DIR = 'metrics'
OUTPUTS_DIR = 'outputs'
MODEL_RESPONSES_DIR = 'model_responses'
EVALUATED_RESPONSES_DIR = 'evaluated_responses'

INITIAL_RESULTS_FILE = 'initial_results'
FT_RESULTS_FILE = 'ft_results'
HALLU_DETECTOR_FILE = 'hallucination_detector.pth'
SKIP_EVAL = True

DATASET_FILE_TRAIN = 'train_gemma.csv'
DATASET_FILE_TEST = 'test_gemma.csv'
DATASET_FILE_EVAL = 'test_gemma.csv'
TRAIN_COL = 'tokenized'
QUERY_COL = 'question'
CONTEXT_COL = 'context'
IDEAL_RESP_COL = 'answer'
DATASET_COL = 'dataset'

MODEL_ID = "unsloth/gemma-2-9b-it-bnb-4bit"
TOKENIZERS_PARALLELISM = 'false'
USE_UNSLOTH = False

SYSTEM_MSG = """
You are an assistant in the systems monitoring team. Your job will be to answer
questions accurately based on the given context. IF YOU DO NOT KNOW THE ANSWER
TO THE QUESTION, RETURN THE ANSWER THAT THE CONTEXT DOES NOT INCLUDE THE GIVEN INFORMATION.
Don't make things up.
Answer only on the basis of the knowledge contained in the context. If the context is incomplete, don't lead conjecture.
Answers should be clear and precise. Pay special attention to the names of applications, services, tools, components
- it is crucial to return information that is consistent for the subject. Context will be given in portions (chunks), your
task is to infer which information is most closely related to the question, even if any portion of the context,
seems to answer the question, always check if this is exactly what the user might have meant. If necessary,
then include context from multiple chunks. Each chunk contains basic information at the beginning.
"""
