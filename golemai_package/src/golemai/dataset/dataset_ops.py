import ast
import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from golemai.config import LOGGER_LEVEL
from golemai.io.file_ops import load_json, read_file_to_df, save_csv
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)


def create_class_mapping(
    df: pd.DataFrame,
    label_column: str,
    output_file: str,
    multi_target: bool = False,
) -> Dict[str, int]:
    """
    Create a class mapping by finding all unique values in the lists in the class column,
    sort them, and assign them an index. Save the mapping to a JSON file.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        label_column (str): Column containing the list of labels.
        output_file (str): Path to the JSON file to save the class mapping.

    Returns:
        Dict[str, int]: Dictionary containing the class mapping.
    """

    # Parse the label lists correctly
    def parse_label_list(label_list):
        if isinstance(label_list, str):
            return eval(label_list)
        return label_list

    df[label_column] = df[label_column].apply(parse_label_list)

    if multi_target:
        # Explode the list of labels to create a long format DataFrame
        exploded_df = df[[label_column]].explode(label_column)

        # Get unique labels and sort them
        unique_labels = sorted(exploded_df[label_column].unique())
    else:
        unique_labels = sorted(df[label_column].unique())

    # Create a mapping from labels to indices
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Save the mapping to a JSON file
    with open(output_file, "w") as f:
        json.dump(label_mapping, f, indent=4, ensure_ascii=False)

    return label_mapping


def map_class_column(
    df: pd.DataFrame,
    label_column: str,
    mapping_file: str,
    multitarget: bool = False,
) -> pd.DataFrame:
    """
    Map the class column based on the class mapping JSON file and convert lists of labels
    into binary lists.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        label_column (str): Column containing the list of labels.
        mapping_file (str): Path to the JSON file containing the class mapping.
        multitarget (bool): Whether the class column contains multiple targets (default is MULTI_TARGET).

    Returns:
        pd.DataFrame: DataFrame with the class column converted to binary lists.
    """
    # Load the class mapping
    class_mapping = load_json(mapping_file)
    num_classes = len(class_mapping)

    if multitarget:

        def convert_to_binary_list(label_list):
            binary_list = [0] * num_classes
            for label in label_list:
                if label in class_mapping:
                    binary_list[class_mapping[label]] = 1
                else:
                    logger.error(f"Label '{label}' not found in class mapping.")
            return binary_list

        df[label_column] = df[label_column].apply(
            lambda x: (
                convert_to_binary_list(eval(x))
                if isinstance(x, str)
                else convert_to_binary_list(x)
            )
        )
    else:
        df[label_column] = df[label_column].apply(
            lambda x: class_mapping[x] if x in class_mapping else -1
        )

    return df


def one_hot_encode(labels: pd.Series, unique_labels: list) -> list:
    """
    One-hot encode a list of labels based on a list of unique labels.

    Args:
        labels (pd.Series): List of labels to encode.
        unique_labels (list): List of unique labels.

    Returns:
        list: One-hot encoded vector of labels.
    """

    if not isinstance(labels, list):
        return None
    # Initialize a vector of zeros with length equal to number of unique labels
    one_hot_vector = [0] * len(unique_labels)

    # Set the corresponding indices to 1 for each label in the input list
    for label in labels:
        if label in unique_labels:
            index = unique_labels.index(label)
            one_hot_vector[index] = 1

    return one_hot_vector


def check_if_present_in_config(
    row: pd.Series, column_name: str, label_list: list
) -> bool:
    """
    Check if all values in a list are present in a given list of labels.

    Args:
        row (pd.Series): Row of the DataFrame.
        column_name (str): Name of the column containing the list of labels.
        label_list (list): List of unique labels.

    Returns:
        bool: True if all values in the list are present in the label list, False otherwise.
    """

    # logger.info(f"check_if_present_in_config: {column_name = }, {label_list = }")

    if pd.isna(row[column_name]):
        return True

    # values = ast.literal_eval(row[column_name])
    values = row[column_name]
    return all(value in label_list for value in values)


def one_hot_encode_column(
    df: pd.DataFrame, column_name: str, unique_labels: list
) -> pd.DataFrame:
    """
    One-hot encode a column with lists of labels based on a list of unique labels.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column containing the list of labels.
        unique_labels (list): List of unique labels.

    Returns:
        pd.DataFrame: DataFrame with the column one-hot encoded.
    """

    logger.info(f"one_hot_encode_column: {column_name = }, {unique_labels = }")

    if not df.apply(
        lambda x: check_if_present_in_config(
            x, column_name, label_list=unique_labels
        ),
        axis=1,
    ).all():
        logger.error(
            f"Column '{column_name}' contains values not present in the unique labels list."
        )
        raise ValueError(
            f"Column '{column_name}' contains values not present in the unique labels list."
        )

    return df[column_name].apply(lambda x: one_hot_encode(x, unique_labels))


def check_list_str(x: str):
    """
    Convert a string representation of a list to a list.
    """

    try:
        ast.literal_eval(x)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False
    else:
        return True


def check_if_multitarget(df: pd.DataFrame, target_col: str | list) -> bool:
    """
    Check if the target column contains multiple targets.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target_col (str): Column containing the target.

    Returns:
        bool: True if the target column contains multiple targets, False otherwise.
    """  # noqa E501

    if isinstance(target_col, (list, tuple)):
        return True

    if (
        df[target_col]
        .apply(lambda x: isinstance(x, (list, np.ndarray, tuple)))
        .all()
    ):
        return True

    if df[target_col].apply(lambda x: not isinstance(x, str)).any():
        return False

    if df[target_col].apply(check_list_str).all():
        return True

    return False


def prepare_target(target_col: str | list, df: pd.DataFrame) -> np.ndarray:
    """
    Prepare the target column for training.

    Args:
        target_col (str | list): Column containing the target.
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        np.ndarray: Prepared target column.
    """

    if isinstance(target_col, list):

        logger.info(f"Multitarget column detected")
        y = df[target_col]

    elif (
        df[target_col]
        .apply(lambda x: isinstance(x, (list, np.ndarray, tuple)))
        .all()
    ):

        logger.info(f"Multitarget column detected")
        y = df[target_col]

    elif not check_if_multitarget(df, target_col=target_col):
        logger.info(f"Single target column detected")
        y = df[target_col]
    else:
        logger.info(f"Multitarget column detected")
        y = df[target_col].apply(ast.literal_eval)

    y = np.array(y.tolist())

    return y


def split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    label_column: str,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing the data to split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        label_column (str): Column containing the binary labels.
        stratify: Whether to perform stratification (default is True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
    """
    if stratify:
        stratify_col = df[label_column].apply(lambda x: sum(x))
    else:
        stratify_col = None

    # Perform the split
    logger.info(
        f"Splitting data with test size = {test_size}, random state = {random_state}, stratify = {stratify}"
    )
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    logger.info(
        f"Data split completed. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    for split_df, split_name in zip([train_df, test_df], ["train", "test"]):
        label_counts = (
            pd.DataFrame(split_df[label_column].tolist()).sum().to_dict()
        )
        logger.info(f"Class distribution in the {split_name} dataset:")
        for label, count in label_counts.items():
            logger.info(f"Class '{label}': {count} samples")

    return train_df, test_df


def train_test_split_multitarget(
    target_array: np.ndarray, test_size: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and test sets. This function is used for multitarget classification.

    Args:
        target_array (np.ndarray): Array containing the target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test indices.
    """

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    flattened_labels = np.argmax(target_array, axis=1)

    for train_index, test_index in sss.split(
        np.zeros(len(flattened_labels)), flattened_labels
    ):
        pass

    return train_index, test_index


def split_dataset(
    input_file: str,
    target_col: str | list = "target",
    output_train_file: str = "train.csv",
    output_test_file: str = "test.csv",
    class_mapping_file: str = "class_mapping.json",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to split the dataset, save the results, and save metadata.

    Args:
        input_file (str): Path to the input CSV file.
        target_col (str | list): Column containing the target (default is 'target').
        output_train_file (str): Path to the output training CSV file (default is 'train.csv').
        output_test_file (str): Path to the output test CSV file (default is 'test.csv').
        class_mapping_file (str): Path to the class mapping JSON file (default is 'class_mapping.json').
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
    """

    logger.info(
        f"split_dataset: {input_file = }, {output_train_file = }, {output_test_file = }, {class_mapping_file = }"
    )

    df = read_file_to_df(input_file)

    df[target_col] = prepare_target(df=df, target_col=target_col)

    if check_if_multitarget(df, target_col):

        train_idx, test_idx = train_test_split_multitarget(
            target_array=np.array(df[target_col].values.tolist()),
            test_size=test_size,
            random_state=random_state,
        )

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    else:

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col],
        )

    for split_df, output_file in zip(
        [train_df, test_df], [output_train_file, output_test_file]
    ):
        save_csv(split_df, output_file)

    return train_df, test_df
