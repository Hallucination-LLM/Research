import ast
import logging

import pandas as pd
from golemai.config import LOGGER_LEVEL
from golemai.enums import JsonSchemaKeys, JsonSchemaTypes
from golemai.io.file_ops import read_file_to_df, save_df_to_file

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)


def validate_columns(
    df: pd.DataFrame,
    text_column: str = "text",
    metadata_column: str = "metadata",
) -> None:
    """
    Validate that required columns exist and metadata is of correct type.
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        text_column (str): The name of the column with text data.
        metadata_column (str): The name of the column with metadata.

    Raises:
        ValueError: If the required columns are missing or metadata is not of type object.
    """

    logger.debug(f"validate_columns: {text_column = }, {metadata_column = }")

    if any(col not in df.columns for col in [text_column, metadata_column]):
        error_msg = (
            f"Missing required columns: {text_column} or {metadata_column}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not pd.api.types.is_object_dtype(df[metadata_column]):
        error_msg = f"The column {metadata_column} is not of type object"
        logger.error(error_msg)
        raise ValueError(error_msg)

    df[metadata_column] = df[metadata_column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    logger.debug("Columns validated successfully")


def check_entity_in_text(
    df: pd.DataFrame,
    text_column: str = "text",
    metadata_column: str = "metadata",
    is_valid_column: str = "is_valid",
) -> None:
    """Check if entities in metadata are present in the corresponding text.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        text_column (str): The name of the column with text data.
        metadata_column (str): The name of the column with metadata.
        is_valid_column (str): The name of the column to store the result of the check.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column indicating if the entities are present in the text.
    """
    logger.debug(
        f"check_entity_in_text: {text_column = }, {metadata_column = }"
    )

    def is_entity_in_text(row: pd.Series) -> int:

        metadata = row[metadata_column]
        text = row[text_column]

        for entity_value in metadata.values():

            if isinstance(entity_value, list):

                if not all(str(item) in text for item in entity_value):
                    return 0

            else:
                if str(entity_value) not in text:
                    return 0
        return 1

    df[is_valid_column] = df.apply(is_entity_in_text, axis=1)
    logger.debug("Entities checked successfully")


def generate_json_schema(
    df: pd.DataFrame,
    metadata_column: str = "metadata",
    json_schema_column: str = "json_schema",
    is_valid_column: str = "is_valid",
    entity_types: dict = None,
    required: dict = None,
) -> None:
    """Generate JSON schema for valid rows.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        metadata_column (str): The name of the column with metadata.
        json_schema_column (str): The name of the column to store the JSON schema.
        is_valid_column (str): The name of the column indicating if the entities are present in the text.
        entity_types (dict): A dictionary mapping entity names to their types.
        required (dict): A dictionary indicating which fields are required.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'json_schema' containing the JSON schema for valid records.
    """

    logger.debug(f"generate_json_schema: {metadata_column = }")

    def create_schema(metadata):
        schema = prepare_json_schema(metadata, entity_types, required)
        return str(schema)

    df[json_schema_column] = df.apply(
        lambda row: (
            create_schema(row[metadata_column])
            if row[is_valid_column] == 1
            else None
        ),
        axis=1,
    )
    logger.debug("JSON schema generated successfully")


def prepare_json_schema(
    x: list | dict,
    entity_types: dict = None,
    required: dict = None,
    title: str = "AnswerFormat",
) -> dict:
    """Prepare a JSON schema for the given data.

    Args:
        x (list | dict): The list of entities or dictionary of entities and their types.
        entity_types (dict): The dictionary of entities and their types.
        required (dict): The dictionary of required entities.
        title (str): The title of the schema.

    Returns:
        dict: The JSON schema for the given data.
    """
    logger.debug(f"prepare_json_schema: {title = }")

    if entity_types is None:
        logger.warning("Entity types not provided. Defaulting to 'string'")
        entity_types = {}

    if required is None:
        logger.warning(
            "Required fields not provided. Defaulting to all fields required"
        )
        required = {}

    schema = {
        JsonSchemaKeys.PROPERTIES.value: {},
        JsonSchemaKeys.REQUIRED.value: [],
        JsonSchemaKeys.TITLE.value: title,
        JsonSchemaKeys.TYPE.value: JsonSchemaTypes.OBJECT.value,
    }

    for entity_name in x:

        ent_type = entity_types.get(entity_name, JsonSchemaTypes.STRING.value)

        schema[JsonSchemaKeys.PROPERTIES.value][entity_name] = {
            JsonSchemaKeys.TITLE.value: " ".join(
                entity_name.split("_")
            ).title(),
            JsonSchemaKeys.TYPE.value: ent_type,
        }

        if ent_type == JsonSchemaTypes.ARRAY:
            schema[JsonSchemaKeys.PROPERTIES.value][entity_name][
                JsonSchemaKeys.ITEMS.value
            ] = {JsonSchemaKeys.TYPE.value: JsonSchemaTypes.STRING.value}

        if required.get(entity_name, True):
            schema[JsonSchemaKeys.REQUIRED.value].append(entity_name)

    logger.debug("JSON schema prepared successfully")
    return schema


def output_json_schema(
    input_file: str,
    output_file: str = "schema.parquet",
    text_column: str = "text",
    metadata_column: str = "metadata",
    entity_types: dict = None,
    required: dict = None,
) -> pd.DataFrame:
    """Main function to process the input data and generate a JSON schema.

    Args:
        input_file (str): The path to the input file containing the data.
        output_file (str): The path to the output file where the processed data will be saved.
                           Defaults to a file named `schema.parquet`.
        text_column (str): The name of the column that contains text data.
                           Defaults to the `text`.
        metadata_column (str): The name of the column that contains metadata (e.g., entities).
                               Defaults to the `metadata`.
        entity_types (dict, optional): A dictionary mapping entity names to their types (e.g., "string", "integer").
                                       If not provided, default types are used.
        required (dict, optional): A dictionary indicating which fields are required (True) and which are optional (False).
                                   If not provided, all fields are considered required by default.

    Returns:
        pd.DataFrame: The processed DataFrame with the JSON schema.
    """

    logger.debug(
        f"output_json_schema: {input_file = }, {output_file = }, {text_column = }, {metadata_column = }, {entity_types = }, {required = }"
    )

    df = read_file_to_df(input_file)
    validate_columns(df, text_column, metadata_column)
    check_entity_in_text(df, text_column, metadata_column)

    generate_json_schema(
        df=df,
        metadata_column=metadata_column,
        entity_types=entity_types,
        required=required,
    )

    save_df_to_file(df, output_file)
    return df
