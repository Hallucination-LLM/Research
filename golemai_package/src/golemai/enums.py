from enum import Enum

import numpy as np


class LocalModelAppKeys(str, Enum):

    USER_INPUT = "user_input"
    SYSTEM_MSG = "system_msg"
    MODEL_RESPONSE = "model_response"
    GENERATED_TEXT = "generated_text"
    MODEL_NAME = "model_name"
    DOCUMENTS_INPUT = "documents_input"
    RERANKING_TOP_N = "top_n"
    RERANKED_DOCS = "reranked_docs"
    DOCUMENTS_ORDER = "documents_order"
    MODEL_ID = "model_name"
    RERANK_SCORES = "rerank_scores"
    JSON_SCHEMA = "json_schema"

    def __str__(self) -> str:
        return str.__str__(self)


class RESTKeys(str, Enum):

    HTTP = "http://"

    def __str__(self) -> str:
        return str.__str__(self)


class AppEngineKeys(str, Enum):
    QA_ENGINE = "qa_engine"
    SEARCH_ENGINE = "search_engine"

    def __str__(self) -> str:
        return str.__str__(self)


class ModelNames(str, Enum):

    MMLW_ROBERTA_LARGE = "sdadas/mmlw-roberta-large"
    MMLW_RETRIEVAL_ROBERTA_LARGE = "sdadas/mmlw-retrieval-roberta-large"
    SBERT_BASE_CASED_PL = "Voicelab/sbert-base-cased-pl"
    TFIDF = "tfidf"

    def __str__(self) -> str:
        return str.__str__(self)


class DFColumnsEnum(str, Enum):
    EMBEDDINGS = "embeddings"
    FILENAME = "filename"
    TEXT = "text"
    CLASS = "class"
    CLASS_MOST_FREQUENT = "class_most_frequent"
    CLASS_LEAST_FREQUENT = "class_least_frequent"

    def __str__(self) -> str:
        return str.__str__(self)


class EmbMetaFields(str, Enum):

    UUID = "uuid"
    DATASET_UUID = "dataset_uuid"
    MODEL_NAME = "model_name"
    TFIDF = "tfidf"
    TIMESTAMP = "timestamp"
    FILENAME = "embeddings_filename"
    DIM = "embeddings_dim"

    def __str__(self) -> str:
        return str.__str__(self)


class TrainMetaFields(str, Enum):

    UUID = "uuid"
    DATASET_UUID = "dataset_uuid"
    EMBEDDINGS_UUID = "embeddings_uuid"
    MODEL_UUID = "model_uuid"
    TIMESTAMP = "timestamp"
    TARGET_COL = "target_col"
    METRICS = "metrics"
    MODEL_MODULE = "model_module"
    MODEL_CLASS = "model_class"
    MODEL_PARAMS = "model_params"
    MODEL_FILENAME = "model_filename"

    def __str__(self) -> str:
        return str.__str__(self)


class DatasetMetaFields(str, Enum):

    UUID = "uuid"
    VERSION = "version"
    INPUT_FILE = "input_file"
    OUTPUT_TRAIN_FILE = "output_train_file"
    OUTPUT_TEST_FILE = "output_test_file"
    TEST_SIZE = "test_size"
    RANDOM_STATE = "random_state"
    STRATIFY = "stratify"
    TRAIN_ROWS = "train_rows"
    TRAIN_COLUMNS = "train_columns"
    TEST_ROWS = "test_rows"
    TEST_COLUMNS = "test_columns"
    TIMESTAMP = "timestamp"

    def __str__(self) -> str:
        return str.__str__(self)


class GenDFColumns(str, Enum):
    TEXT_COL = "text"
    FILENAME_COL = "file_name"
    METADATA_COL = "metadata"
    HASH_COL = "hash"
    IS_VALID_COL = "isValid"
    IS_OCR_COL = "isOCRError"
    TEMPLATE_COL = "template"

    def __str__(self) -> str:
        return str.__str__(self)


class DictKeys(str, Enum):
    PATH_TO_TEMPLATES = "path_to_templates"
    DOCUMENT_TYPES = "document_types"
    FILES = "files"
    FILENAME = "filename"
    HASH = "hash"
    ENTITIES = "entities"
    GENERATION_SETTINGS = "generation_settings"
    NUMBER_TO_GENERATE = "number_to_generate"
    TYPE = "type"
    TEXT = "text"
    PROBABILITY_OF_OCR = "probability_to_generate_OCR_Error_rows"
    MAX_CHARACTERS_TO_CHANGE = "max_chars_to_change"
    TIMESTAMP = "timestamp"

    def __str__(self) -> str:
        return str.__str__(self)


class PossibleValuesKeys(str, Enum):
    PODSTAWA_PRAWNA = "podstawa_prawna"
    KONCOWKI_PLN = "koncowki_pln"
    KONCOWKI_EURO = "koncowki_euro"
    FORMA_POMOCY = "forma_pomocy"
    SAMOCHODY = "samochody"
    UBEZPIECZYCIELE = "ubezpieczyciele"
    ORGAN_TOZSAMOSCI = "organ_tozsamosci"
    KM = "km"
    AKCESORIA_SAMOCHODOWE = "akcesoria_samochodowe"
    ELEMENTY_EKSPLOATACYJNE = "elementy_eksploatacyjne"
    STAN_SAMOCHODU = "stan_samochodu"
    RODZAJ_TOZSAMOSCI = "rodzaj_tozsamosci"
    SPOSOB_TRANSAKCJI = "sposob_transakcji"

    def __str__(self) -> str:
        return str.__str__(self)


class DefaultFiles(str, Enum):
    TEMPLATES_JSON = "templates.json"
    USER_CONFIG_YAML = "user_config.yaml"
    ENTITIES_JSON = "entities.json"
    DEFAULT_POSSIBLE_ENTITIES_JSON = "default_possible_entities.json"
    GENERATED_DATASET = "generated_dataset"

    def __str__(self) -> str:
        return str.__str__(self)


class DefaultFolders(str, Enum):
    TXT = "txt"
    SAVE_FOLDER = "output"

    def __str__(self) -> str:
        return str.__str__(self)


class OutputFormats(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

    def __str__(self) -> str:
        return str.__str__(self)


class DefaultValues(int, Enum):
    MIN_PRODUCTION_YEAR = 1980
    MAX_PRODUCTION_YEAR = 2022

    def __int__(self) -> int:
        return super().__int__()


class LocalizationPredictorSuffixes(str, Enum):
    GROUND_TRUTH = "_GT"
    PREDICTION = "_PRED"
    PER_DOCUMENT = "_DOC"
    GLOBAL = "_GLOBAL"
    ZERO_ONE = "_01"
    ENTITIES = "_ENTITIES"
    INDICES = "_INDICES"
    VALUES = "_VALUES"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorPrefixes(str, Enum):
    LOCALIZATION = "LOC_"
    NER_CORECTNESS = "NER_"
    GROUND_TRUTH = "GT_"
    PREDICTION = "PRED_"
    EMBEDDING = "EMB_"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorColumns(str, Enum):
    LEVENSTEIHN_DISTANCE = "levensteihn_distance"
    NORMALIZED_LEVENSTEIHN_DISTANCE = "normalized_levensteihn_distance"
    SPARSE_VECTOR = "sparse_vector"
    INDICES = "indices"
    EMBEDDING_INDICES = "indices"
    EMBEDDING_VALUES = "values"
    INSTANCE_ID = "instance_ID"  # used as ID of document
    SUB_INSTANCE_ID = "sub_instance_ID"  # if an entity has multiple values (is a list or a name like Jan Krzysztof), they are stored in different rows
    GROUND_TRUTH = "ground_truth"
    PREDICTION = "prediction"
    VALUE = "value"
    VALUES = "values"
    VARIABLE = "variable"
    ENTITY = "entity"
    EMPTY_STRING = ""
    COSINE_SIMMILARITY = "cosine_simmilarity"
    LEVENSTEIHN_DISTANCE_PROBABILITY = "levensteihn_distance_probability"
    LEVENSTEIHN_DISTANCE_HEURISTIC_THRESHOLD = (
        "levensteihn_distance_heuristic_threshold"
    )
    COSINE_SIMMILARITY_HEURISTIC_THRESHOLD = (
        "cosine_simmilarity_heuristic_threshold"
    )
    Y_TRUE_TRAIN = "y_true_train"
    LOCALIZATION_ACCURACY = "localization_accuracy"
    HALLUCINATED_ENTITIES = "hallucinated_entities"
    HALLUCINATED_ENTITIES_PERCENTAGE = "hallucinated_entities_percentage"
    INCORRECT_SYMBOLS = "incorrect_symbols"
    INCORRECT_SYMBOLS_PERCENTAGE = "incorrect_symbols_percentage"
    TOTAL_SYMBOLS = "total_symbols"
    TOTAL_ENTITIES = "total_entities"
    EXACT_MATCH = "exact_match"
    UNIQUE_ENTITIES = "unique_entities"
    UNIQUE_ENTITIES_X = "unique_entities_x"
    UNIQUE_ENTITIES_Y = "unique_entities_y"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorDictNames(str, Enum):
    EMBEDDING_PREDICTIONS = "embedding_predictions"
    EMBEDDING_GROUND_TRUTH = "embedding_ground_truth"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorThreholdTypes(str, Enum):
    LEVENSTEIHN = "Levensteihn"
    COSINE = "Cosine"
    DEFAULT = "Default"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorEmbeddingEndpoint(str, Enum):
    EMBEDDING_HOST = "172.16.2.203"
    EMBEDDING_PORT = "3013"
    EMBEDDING_PATH = "sparse"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorEmbeddingSources(str, Enum):
    API = "api"
    MODEL = "model"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorDefaultFloats(float, Enum):
    DEFAULT_THRESHOLD = 0.5
    THRESHOLD_COEFFICIENT = 0.5

    def __float__(self) -> float:
        return super().__float__()


class LocalizationPredictorDefaultInts(int, Enum):
    ORG_EMBEDDING_SHAPE = 30522
    BATCH_SIZE = int(256)
    EMBEDDING_BATCH_SIZE = int(32)
    MAX_TEXT_LENGTH_LOCAL_EMBEDDING = 512

    def __int__(self) -> int:
        return super().__int__()


class LocalizationPredictorPredictProbaModes(str, Enum):
    LEVENSTEIHN = "leven"
    COSINE = "cosine"
    FIT = "fit"

    def __str__(self) -> str:
        return super().__str__()


class LocalizationPredictorFillValues(float, Enum):
    COSINE_SIMMILARITY_AND = 1
    LEVENSTEIHN_DISTANCE_PROBABILITY_AND = 1
    LEVENSTEIHN_DISTANCE_AND = 0
    NORMALIZED_LEVENSTEIHN_DISTANCE_AND = 0

    LEVENSTEIHN_DISTANCE_XOR = (np.nan,)
    COSINE_SIMMILARITY_XOR = (0,)
    LEVENSTEIHN_DISTANCE_PROBABILITY_XOR = (0,)
    NORMALIZED_LEVENSTEIHN_DISTANCE_XOR = (np.nan,)

    COSINE_SIMMILARITY_EXACT_MATCH = 1
    LEVENSTEIHN_DISTANCE_PROBABILITY_EXACT_MATCH = 1
    LEVENSTEIHN_DISTANCE_EXACT_MATCH = 0
    NORMALIZED_LEVENSTEIHN_DISTANCE_EXACT_MATCH = 0

    def __float__(self) -> float:
        return super().__float__()


class AutencoderReconKeys(Enum):
    MIN_RECON_ERROR = "min_reconstruction_error"
    MAX_RECON_ERROR = "max_reconstruction_error"

    def __str__(self) -> str:
        return str.__str__(self)


class JsonSchemaKeys(str, Enum):

    PROPERTIES = "properties"
    REQUIRED = "required"
    TITLE = "title"
    TYPE = "type"
    ITEMS = "items"

    def __str__(self) -> str:
        return str.__str__(self)


class JsonSchemaTypes(str, Enum):

    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"

    def __str__(self) -> str:
        return str.__str__(self)


class EntityValidationDocumentTypes(str, Enum):
    DEMINIMIS = "deminimis"
    UMOWAKUPNASPRZEDARZYSAMOCHODU = "samochody"


class OcrEvaluationResultsKeys(str, Enum):
    CORRECT_CHARS_RATIO_ALL_TEXT = "correct_chars_ratio_all_text"
    CORRECT_CHARS_RATIO_CONTENT = "correct_chars_ratio_content"
    CORRECT_WORDS_RATIO_CONTENT = "correct_words_ratio_content"
    CORRECT_CHARS_RATIO_NUMS = "correct_chars_ratio_nums"
    CORRECT_CHARS_RATIO_CUSTOM_NUMS = "correct_chars_ratio_custom_nums"
    CORRECT_WORDS_RATIO_CUSTOM_NUMS = "correct_words_ratio_custom_nums"

    def __str__(self) -> str:
        return str.__str__(self)


class StatsKeys(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"

    def __str__(self) -> str:
        return str.__str__(self)
