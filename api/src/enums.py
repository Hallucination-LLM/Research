from enum import Enum;

class Metadata(str, Enum):
    META_AREA = 'obszar'
    META_FILE = 'plik'
    META_PATH = 'sciezka'
    META_TEXT = 'text'
    META_TABLE = 'tables'
    META_PHONE = 'phone'
    META_EMAIL = 'email'
    
    ID = 'id'

    META_MODEL = 'model_embed'
    META_TOKEN_LIMIT = 'token_limit'

    META_VERSION = 'wersja'
    META_IS_TABLE = 'is_table'
    
    def __str__(self) -> str:
        return str.__str__(self)
    
class TableDict(str, Enum):
    PARAGRAPHS = 'PARAGRAPHS'
    CONTENT = 'CONTENT'
    CONTEXT = 'CONTEXT'
    SPLITTER = '//+//'

    def __str__(self) -> str:
        return str.__str__(self)

class Splits(str, Enum):
    SPLIT_SIGN = '//'
    SPLIT_TITLE = f'{SPLIT_SIGN}+{SPLIT_SIGN}'
    APPEND_CHUNK = f'{SPLIT_SIGN}-{SPLIT_SIGN}'
    INDEX_SPLIT = f'{SPLIT_SIGN}#{SPLIT_SIGN}'
    
    TEXT_SPLIT = '//TEXT://'
    CHAPTERS_SPLIT = '//CHAPTERS://'
    
    def __str__(self) -> str:
        return str.__str__(self)
    
class DomainNames(str, Enum):

    ARCHITEKTURA = 'Architektura'

    def __str__(self) -> str:
        return str.__str__(self)
    
class LLMConfig(str, Enum):

    DEFAULT_LLM = 'local'
    
    def __str__(self) -> str:
        return str.__str__(self)
    
class LocalModelAppKeys(str, Enum):

    USER_INPUT = 'user_input'
    SYSTEM_MSG = 'system_msg'
    MODEL_RESPONSE = 'model_response'
    GENERATED_TEXT = 'generated_text'
    MODEL_NAME = 'model_name'
    DOCUMENTS_INPUT = 'documents_input'
    RERANKING_TOP_N = 'top_n'
    RERANKED_DOCS = 'reranked_docs'
    DOCUMENTS_ORDER = 'documents_order'
    STREAM_RESPONSE = 'stream_response'
    JSON_SCHEMA = 'json_schema'
    
    def __str__(self) -> str:
        return str.__str__(self)
    
class DefaultLocalModels(str, Enum):

    EMBEDDING_MODEL = 'sdadas/mmlw-retrieval-roberta-large'
    RERANKER_MODEL = 'sdadas/polish-reranker-large-ranknet'

    def __str__(self) -> str:
        return str.__str__(self)
    
class LocalModelsNamesKeys(str, Enum):

    EMBEDDING_MODEL = 'embedding_model'
    RERANKER_MODEL = 'reranker_model'
    LLM_QA_MODEL = 'llm_qa'
    
    def __str__(self) -> str:
        return str.__str__(self)
