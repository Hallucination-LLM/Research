import json
import logging
import os
import pickle
import uuid
import numpy as np
from itertools import islice
from typing import Any, Dict, Iterable

import pandas as pd
import yaml
from golemai.config import LOGGER_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)


def read_file_to_df(
    filepath: str, usecols: list = None, delimiter: str = ",", nrows: int = None
) -> pd.DataFrame:
    """
    Read a file into a pandas DataFrame.

    Args:
        filepath (str): The path to the file to read.
        usecols (list, optional): The columns to read. Defaults to None.
        delimiter (str, optional): The delimiter to use. Defaults to ",".
        nrows (int, optional): The number of rows to read. Defaults to None.

    Returns:
        pd.DataFrame: The loaded data.
    """

    logger.debug(f"read_file: {filepath = }")

    ALLOWED_EXTENSIONS = {
        ".csv": lambda x: pd.read_csv(
            x, usecols=usecols, delimiter=delimiter, nrows=nrows
        ),
        ".json": lambda x: pd.read_json(x, nrows=nrows),
        ".parquet": lambda x: pd.read_parquet(x, columns=usecols),
        ".pickle": lambda x: pd.read_pickle(x),
        ".xlsx": lambda x: pd.read_excel(x, usecols=usecols, nrows=nrows),
        ".xls": lambda x: pd.read_excel(x, usecols=usecols, nrows=nrows),
        ".table": lambda x: pd.read_table(
            x, usecols=usecols, delimiter=delimiter, nrows=nrows
        ),
        ".txt": lambda x: pd.read_csv(
            x, usecols=usecols, delimiter=delimiter, nrows=nrows
        ),
    }

    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    _, ext = os.path.splitext(filepath)

    if ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}")

    df = ALLOWED_EXTENSIONS[ext](filepath)
    logger.debug(f"Data loaded successfully with shape: {df.shape}")

    return df


def save_df_to_file(
    df: pd.DataFrame, filepath: str, delimiter: str = ",", index: bool = False
):
    """
    Save a pandas DataFrame to a file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The path to the file where data should be saved.
        delimiter (str): Delimiter to use for CSV/TXT files. Default is ','.
        index (bool): Whether to write row names (index). Default is False.
    """

    logger.debug(f"save_df_to_file: {filepath = }")

    ALLOWED_EXTENSIONS = {
        ".csv": lambda x: df.to_csv(x, sep=delimiter, index=index),
        ".json": lambda x: df.to_json(x, orient="records", lines=True),
        ".parquet": lambda x: df.to_parquet(x, index=index),
        ".pickle": lambda x: df.to_pickle(x),
        ".xlsx": lambda x: df.to_excel(x, index=index),
        ".xls": lambda x: df.to_excel(x, index=index),
        ".table": lambda x: df.to_csv(x, sep=delimiter, index=index),
        ".txt": lambda x: df.to_csv(x, sep=delimiter, index=index),
    }

    _, ext = os.path.splitext(filepath)

    if ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported file format: {ext}")
        raise ValueError(f"Unsupported file format: {ext}")

    # Execute the appropriate saving function based on file extension
    ALLOWED_EXTENSIONS[ext](filepath)
    logger.debug(f"Data saved successfully to {filepath}")


def load_json(file_path: str) -> dict:
    """
    Load a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """

    logger.debug(f"Loading JSON file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"JSON file loaded successfully.")

    return data


def save_json(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to save the JSON file.
    """

    logger.debug(f"Saving JSON file: {file_path}")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logger.debug(f"JSON file saved successfully.")


def load_pickle(file_path: str) -> any:
    """
    Load a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Any: The loaded pickle data.
    """

    logger.debug(f"load_pickle: {file_path = }")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data


def save_pickle(data: any, file_path: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data (any): The data to save.
        file_path (str): The path to save the pickle file.
    """

    logger.debug(f"save_pickle: {file_path = }")

    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    logger.debug(f"Data saved successfully to {file_path}")

def save_numpy(data: any, file_path: str) -> None:
    """
    Save data to a numpy file.

    Args:
        data (any): The data to save.
        file_path (str): The path to save the numpy file.
    """

    logger.debug(f"save_numpy: {file_path = }")

    with open(file_path, "wb") as f:
        np.save(f, data)

    logger.debug(f"Data saved successfully to {file_path}")


def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The loaded YAML data.
    """

    logger.debug(f"load_yaml: {file_path = }")

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None

    logger.debug(f"YAML file loaded successfully.")

    return data


def save_yaml(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to save the YAML file.
    """

    logger.debug(f"save_yaml: {file_path = }")

    with open(file_path, "w") as f:
        yaml.dump(data, f)

    logger.debug(f"YAML file saved successfully.")


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the data to save.
        file_path (str): Path to the CSV file to save the data.
    """
    logger.debug(f"Saving data to {file_path}")
    df.to_csv(file_path, index=False)
    logger.debug(f"Data saved successfully to {file_path}")


def save_metadata(metadata: Dict[str, Any], metadata_dir: str) -> None:
    """
    Save metadata to a JSON file named with its UUID, handling versioning to avoid overwriting.

    Args:
        metadata (dict): Metadata to save.
        metadata_dir (str): Directory to save the metadata JSON file.
    """  # noqa: E501
    uuid = metadata.get("uuid")
    if not uuid:
        logger.error("UID not found in metadata. Cannot save metadata.")
        return

    version = metadata.get("version", 1)
    base_filename = f"{uuid}.json"
    metadata_file = os.path.join(metadata_dir, base_filename)

    # Check if the file already exists and handle versioning
    while os.path.exists(metadata_file):
        # Find the last version
        files = [
            f for f in os.listdir(metadata_dir) if f.startswith(f"{uuid}__V")
        ]
        if files:
            last_version = max(
                int(f.split("__V")[-1].split(".json")[0]) for f in files
            )
            version = last_version + 1
        metadata["version"] = version
        metadata_file = os.path.join(metadata_dir, f"{uuid}__V{version}.json")

    logger.debug(f"Saving metadata to {metadata_file}")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.debug("Metadata saved successfully.")


def get_next_uuid(metadata_dir: str) -> str:
    """
    Get the next unique identifier (uuid) for the metadata.

    Args:
        metadata_dir (str): Directory containing the metadata JSON files.

    Returns:
        str: The next uuid as a string.
    """
    try:
        existing_files = [
            f for f in os.listdir(metadata_dir) if f.endswith(".json")
        ]
        existing_uuids = [f.split("__")[0] for f in existing_files]
        new_uuid = str(uuid.uuid4())
        while new_uuid in existing_uuids:
            new_uuid = str(uuid.uuid4())
        return new_uuid
    except FileNotFoundError:
        return str(uuid.uuid4())


def batch_text_data(iterable: Iterable, n: int) -> Iterable:
    """Batch data into tuples of length n. The last batch may be shorter.

    Args:
        iterable (Iterable): The data to batch.
        n (int): The batch size.

    Returns:
        Iterable: An iterable of tuples of length n.
    """

    if n < 1:
        raise ValueError("n must be at least one")

    it = iter(iterable)

    while batch := tuple(islice(it, n)):
        yield batch


def decode_stream(stream, stop_tokens: str = ["<eos>", "<end_of_turn>"]):
    for s in stream:
        try:
            s = s.decode("utf-8")
        except UnicodeDecodeError:
            yield " "
        else:
            if s.strip() in stop_tokens:
                break
            else:
                yield s
