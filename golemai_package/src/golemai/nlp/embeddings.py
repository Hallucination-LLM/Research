from typing import List, Union

import numpy as np
import requests


def get_local_sparse(
    host: str,
    query: str | List[str],
    restore_original: bool = True,
    org_shape: int = 30522,
) -> Union[None, tuple[List[float], str]]:
    """
    Sends a request to a local model to generate sparse embeddings.
    At first the sparse embeddings are in Pinecone format, which is a dictionary with `indices` and `values`.

    Args:
        host (str): Host URL for the local model.
        query (str): User input query.
        restore_original (bool): Whether to restore the original shape of the sparse vector.
        org_shape (int): Original shape of the sparse vector.

    Returns:
        Union[None, tuple[List[float], str]]: Sparse embeddings and model ID.
    """

    response = requests.post(f"http://{host}", json={"user_input": query})

    if not response.ok:
        return None

    response_json = response.json()
    model_response = response_json.get("model_response")

    if restore_original:

        if not isinstance(model_response, list):
            model_response = [model_response]

        restored_vectors = []

        for response in model_response:

            indices = response["indices"]
            values = response["values"]

            restored_vector = np.zeros(org_shape)
            restored_vector[indices] = values

            restored_vectors.append(restored_vector.tolist())

        return restored_vectors

    return model_response


def get_local_embeddings(
    host: str, query: str
) -> Union[None, tuple[List[float], str]]:
    """
    Sends a request to a local model to generate dense embeddings.

    Args:
        host (str): Host URL for the local model.
        query (str): User input query.

    Returns:
        Union[None, tuple[List[float], str]]: Dense embeddings and model ID.
    """

    response = requests.post(f"http://{host}", json={"user_input": query})

    if not response.ok:
        return None

    response_json = response.json()
    model_response = response_json.get("model_response")

    return model_response
