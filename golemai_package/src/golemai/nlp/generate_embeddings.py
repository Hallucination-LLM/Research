import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from golemai.config import LOGGER_LEVEL
from golemai.enums import DFColumnsEnum
from golemai.io.file_ops import load_pickle, read_file_to_df
from golemai.logging.log_formater import init_logger
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = init_logger(LOGGER_LEVEL)


def load_model(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Load the Hugging Face model and tokenizer.

    Args:
        model_name (str): Name of the Hugging Face model to load.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: The loaded model and tokenizer.
    """

    logger.debug(f"Loading model and tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Running model on device: {device}")

    model.to(device)

    logger.debug("Model and tokenizer loaded successfully.")
    return model, tokenizer


def get_embeddings(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the specified model and tokenizer.

    Args:
        model (transformers.PreTrainedModel): Hugging Face model to use.
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer to use.
        texts (List[str]): List of texts to generate embeddings for.
        device (torch.device): Device to run the model on.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        np.ndarray: Generated embeddings.
    """  # noqa E501
    logger.debug("Generating embeddings for batch of texts.")

    try:

        inputs = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
        )

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        logger.debug("Embeddings generated successfully.")
        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.zeros((len(texts), model.config.hidden_size))


def prepare_embeddings(
    embeddings: np.ndarray | list,
    indicies: np.ndarray = None,
    additional_features: np.ndarray | list = None,
) -> np.ndarray:
    """
    Prepare embeddings for training. It can also include additional features and
    select specific indicies.

    Args:
        embeddings (np.ndarray | list): Embeddings to prepare.
        indicies (np.ndarray, optional): Indicies to select from embeddings. Defaults to None.
        additional_features (np.ndarray | list, optional): Additional features to include. Defaults to None.

    Returns:
        np.ndarray: Prepared embeddings.
    """  # noqa E501

    logger.debug(f"prepare_embeddings: {len(embeddings) = }")

    embeddings = np.array(embeddings)

    if indicies is not None:

        embeddings = embeddings[indicies]
        if len(embeddings) != len(indicies):
            logger.error(
                f"Embeddings length mismatch: {embeddings.shape = }, "
                f"{indicies.shape = }"
            )
            return None

    if additional_features is not None:

        embeddings = np.vstack(
            [
                embeddings,
                additional_features.values.reshape(
                    -1, len(additional_features)
                ),
            ]
        )

    logger.debug(f"Embedding loaded successfully: {embeddings.shape = }")
    return embeddings


def load_embeddings(
    embeddings_filepath: str,
    indicies: np.ndarray = None,
    additional_features: pd.DataFrame | pd.Series = None,
) -> np.ndarray:
    """
    Load embeddings from a pickle file and prepare them for training.

    Args:
        embeddings_filepath (str): Path to the embeddings pickle file.
        indicies (np.ndarray): Indicies to select from embeddings.
        additional_features (pd.DataFrame | pd.Series, optional): Additional features to include. Defaults to None.

    Returns:
        np.ndarray: Prepared embeddings.
    """  # noqa E501

    logger.debug(f"load_embeddings: {embeddings_filepath = }")

    embeddings = load_pickle(embeddings_filepath)

    if embeddings is None:
        logger.warning(f"Embeddings file not found: {embeddings_filepath}")
        return None

    embeddings = prepare_embeddings(
        embeddings=embeddings,
        indicies=indicies,
        additional_features=additional_features,
    )

    if embeddings is None:
        logger.warning(f"Error preparing embeddings: {embeddings_filepath}")
        return None

    return embeddings


def generate_tfidf_embeddings(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    max_features: int = 1000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generates TF-IDF embeddings for a list of texts.

    Args:
        train_texts (List[str]): List of texts to fit the TF-IDF vectorizer on and transform.
        test_texts (List[str], optional): List of texts to transform using the fitted vectorizer. Defaults to None.
        max_features (int, optional): Maximum number of features to use in the TF-IDF vectorizer. Defaults to MAX_FEATURES.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Generated TF-IDF embeddings for training and test data.
    """  # noqa E501

    logger.debug("Generating TF-IDF embeddings.")

    vectorizer = TfidfVectorizer(max_features=max_features)
    train_embeddings = vectorizer.fit_transform(train_texts).toarray()
    test_embeddings = (
        vectorizer.transform(test_texts).toarray() if test_texts else None
    )

    logger.debug("TF-IDF embeddings generated successfully.")
    return train_embeddings, test_embeddings


def save_embeddings(embeddings: np.ndarray, output_pickle: str) -> None:
    """
    Save embeddings to a pickle file.

    Args:
        embeddings (np.ndarray): Embeddings to save.
        output_pickle (str): Path to the pickle file.
    """
    logger.debug(f"Saving embeddings to {output_pickle}")
    with open(output_pickle, "wb") as f:
        pickle.dump(embeddings, f)
    logger.debug("Embeddings saved successfully.")


def generate_embeddings(
    output_pickle: str,
    model_name: str = None,
    text_column: str = "text",
    additional_columns: List[str] = None,
    tf_idf: bool = False,
    max_length: int = 512,
    input_file: str = os.path.join("data", "full_data_extra.csv"),
    metadata_dir: str = "metadata",
    max_features: int = 1000,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, str]:
    """
    Generates embeddings for text data in a CSV file using a specified Hugging Face model and/or TF-IDF.

    Args:
        output_pickle (str): Path to save the generated embeddings.
        model_name (str, optional): Name of the Hugging Face model to use. Defaults to None.
        text_column (str, optional): Name of the text column in the input CSV file. Defaults to "text".
        additional_columns (List[str], optional): Additional columns to include in the output DataFrame. Defaults to None.
        tf_idf (bool, optional): Whether to generate TF-IDF embeddings. Defaults to False.
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        input_file (str, optional): Path to the input CSV file. Defaults to "data/full_data_extra.csv".
        metadata_dir (str, optional): Directory to save metadata files. Defaults to "metadata".
        max_features (int, optional): Maximum number of features to use in the TF-IDF vectorizer. Defaults to 1000.
        batch_size (int, optional): Batch size for generating embeddings. Defaults to 32.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the DataFrame with embeddings and the path to the saved pickle file.
    """  # noqa E501

    df = read_file_to_df(input_file)

    if additional_columns is None:
        additional_columns = []

    all_embeddings = []

    if model_name == "tfidf":
        model_name = None
        tf_idf = True

    # Generate model embeddings if model_name is provided
    if model_name:
        logger.debug(f"Starting embedding generation with model: {model_name}")
        model, tokenizer = load_model(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        texts = df[text_column].tolist()

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc=f"Generating {model_name} embeddings",
        ):
            batch_texts = texts[i : i + batch_size]
            embeddings = get_embeddings(
                model, tokenizer, batch_texts, device, max_length
            )
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        logger.debug("Model embeddings generated successfully.")

    # Generate TF-IDF embeddings if tf_idf is True
    if tf_idf:
        logger.debug("Generating TF-IDF embeddings.")
        texts = df[text_column].tolist()
        tfidf_embeddings, _ = generate_tfidf_embeddings(
            texts, max_features=max_features
        )
        if len(all_embeddings) > 0:
            all_embeddings = np.hstack((all_embeddings, tfidf_embeddings))
            logger.debug("TF-IDF embeddings concatenated successfully.")
        else:
            all_embeddings = tfidf_embeddings
            logger.debug("TF-IDF embeddings generated successfully.")

    # Ensure all_embeddings is not empty
    if len(all_embeddings) == 0:
        error_message = (
            "No embeddings were generated. Please specify either a model_name "
            "or set tf_idf to True."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Ensure all_embeddings is a NumPy array
    all_embeddings_array = np.array(all_embeddings)
    df[DFColumnsEnum.EMBEDDINGS] = all_embeddings_array.tolist()

    # Save the embeddings to a pickle file
    save_embeddings(all_embeddings_array, output_pickle)

    logger.debug("Embeddings generation completed successfully.")

    return df, output_pickle
