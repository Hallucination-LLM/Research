# pylint: disable=invalid-name

from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from golemai.config import LOGGER_LEVEL
from golemai.enums import LocalizationPredictorColumns as Columns
from golemai.enums import LocalizationPredictorDefaultFloats as DefaultFloats
from golemai.enums import LocalizationPredictorDefaultInts as DefaultInts
from golemai.enums import (
    LocalizationPredictorEmbeddingEndpoint as EmbeddingEndpoint,
)
from golemai.enums import (
    LocalizationPredictorEmbeddingSources as EmbeddingSources,
)
from golemai.enums import LocalizationPredictorFillValues as FillValues
from golemai.enums import (
    LocalizationPredictorPredictProbaModes as PredictProbaModes,
)
from golemai.enums import LocalizationPredictorSuffixes as Suffixes
from golemai.logging.log_formater import init_logger
from golemai.nlp.embeddings import get_local_sparse
from rapidfuzz.distance import Levenshtein
from tqdm.notebook import tqdm

logger = init_logger(LOGGER_LEVEL)


class NerPredictor(ABC):
    def __init__(
        self,
        prediction_column: str = "prediction",
        ground_truth_column: str = "ground_truth",
        embedding_source: str = "api",
        embedding_url: str = None,
        embedding_model: Any = None,
    ):

        self.prediction_column = prediction_column
        self.ground_truth_column = ground_truth_column
        self.X = None
        self.y = None
        self.embedding_batch_size = DefaultInts.EMBEDDING_BATCH_SIZE
        self.batch_size = DefaultInts.BATCH_SIZE
        self.embedding_source = embedding_source
        self.threshold = DefaultFloats.DEFAULT_THRESHOLD

        # check that embedding_source is either 'api' or 'model'
        if self.embedding_source == EmbeddingSources.API:
            if embedding_url is not None:
                self.embedding_url = embedding_url
            else:
                self.embedding_url = f"{EmbeddingEndpoint.EMBEDDING_HOST}:{EmbeddingEndpoint.EMBEDDING_PORT}/{EmbeddingEndpoint.EMBEDDING_PATH}"
        elif (
            self.embedding_source == EmbeddingSources.MODEL
            and embedding_model is not None
        ):
            # only import SentenceTransformer if embedding_model is provided as it imports slowly
            # pylint: disable=import-outside-toplevel
            from sentence_transformers import SentenceTransformer

            # ensure is SentenceTransformer
            assert (
                isinstance(embedding_model, SentenceTransformer) is True
            ), "Embedding model must be an instance of SentenceTransformer"
            self.embedding_model = embedding_model

    def _calculate_exact_match(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        include_nan: bool = True,
    ) -> np.ndarray:
        """
        Calculate the exact match between the prediction and the ground truth.

        Args:
            X (Union[pd.DataFrame, pd.Series]): Prediction
            y (Union[pd.DataFrame, pd.Series]): Ground truth
            include_nan (bool): Include NaN values in the calculation

        Returns:
            np.ndarray: Indices of the rows where X == y
        """
        logger.debug("Calculating exact match indices")
        if include_nan:
            return np.where((X == y) | (X.isna() & y.isna()))[0]
        else:
            return np.where((X == y))[0]

    def _calculate_non_exact_match_indices(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
    ) -> np.ndarray:
        """
        Calculate the indices of the rows where the prediction is not exact.

        Args:
            X (Union[pd.DataFrame, pd.Series]): Prediction
            y (Union[pd.DataFrame, pd.Series]): Ground truth

        Returns:
            np.ndarray: Indices of the rows where X != y and not both are NaN
        """
        logger.debug("Calculating non-exact match indices")
        return np.where((X != y) & ~(pd.isna(X) & pd.isna(y)))[0]

    def _calculate_xor_nan_indices(
        self, X: pd.Series, y: pd.Series
    ) -> np.ndarray:
        """
        Calculate the indices of the rows where either the prediction or the ground truth is NaN.

        Args:
            X (pd.Series): Prediction
            y (pd.Series): Ground truth

        Returns:
            np.ndarray: Indices of the rows where the prediction is NaN
        """
        logger.debug("Calculating NaN indices using XOR")
        return np.where(X.isna() ^ y.isna())[0]

    def _calculate_and_nan_indices(
        self, X: pd.Series, y: pd.Series
    ) -> np.ndarray:
        """
        Calculate the indices of the rows where both the prediction and the ground truth is NaN.

        Args:
            X (pd.Series): Prediction
            y (pd.Series): Ground truth

        Returns:
            np.ndarray: Indices of the rows where the prediction is NaN
        """
        logger.debug("Calculating NaN indices using AND")
        return np.where(X.isna() & y.isna())[0]

    def _merge_x_y(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the DataFrames X and y.

        Args:
            X (pd.DataFrame): DataFrame X
            y (pd.DataFrame): DataFrame y

        Returns:
            pd.DataFrame: Merged DataFrame
        """
        logger.debug("Merge the DataFrames")

        return pd.merge(
            X,
            y,
            on=[Columns.INSTANCE_ID, Columns.SUB_INSTANCE_ID, Columns.ENTITY],
            suffixes=(Suffixes.PREDICTION, Suffixes.GROUND_TRUTH),
            how="outer",
        )

    def _preprocess_merged_data(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the merged DataFrame.

        Args:
            table (pd.DataFrame): DataFrame with the prediction and the ground truth

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """

        logger.debug("Sort the DataFrame")
        # sort by Columns.INSTANCE_ID, Columns.SUB_INSTANCE_ID, Columns.ENTITY
        table = table.sort_values(
            [
                Columns.INSTANCE_ID,
                Columns.ENTITY,
                Columns.SUB_INSTANCE_ID,
            ]
        ).reset_index(drop=True)

        # replace all NaN with np.nan
        logger.debug("Replace NaN with np.nan")
        table = table.fillna(np.nan)

        return table

    def _set_metrics_for_exact_matches(
        self,
        table: pd.DataFrame,
        indices: np.array,
    ) -> pd.DataFrame:
        """
        Set the values for exact matches.

        Args:
            table (pd.DataFrame): DataFrame with the prediction and the ground truth
            indices (np.array): Indices of the rows where the prediction is equal to the ground truth

        Returns:
            pd.DataFrame: DataFrame with the values set for exact matches
        """
        columns = {
            Columns.COSINE_SIMMILARITY: FillValues.COSINE_SIMMILARITY_EXACT_MATCH,
            Columns.LEVENSTEIHN_DISTANCE: FillValues.LEVENSTEIHN_DISTANCE_EXACT_MATCH,
            Columns.NORMALIZED_LEVENSTEIHN_DISTANCE: FillValues.NORMALIZED_LEVENSTEIHN_DISTANCE_EXACT_MATCH,
            Columns.LEVENSTEIHN_DISTANCE_PROBABILITY: FillValues.LEVENSTEIHN_DISTANCE_PROBABILITY_EXACT_MATCH,
        }

        logger.debug("Set the values for exact matches")

        for column, value in columns.items():
            table.loc[indices, column] = value
        return table

    def _set_metrics_for_xor_nan_values(
        self,
        table: pd.DataFrame,
        nan_indices: np.array,
    ) -> pd.DataFrame:
        """
        Set the values for NaN values.

        Args:
            table (pd.DataFrame): DataFrame with the prediction and the ground truth
            nan_indices (np.array): Indices of the rows where the prediction is NaN

        Returns:
            pd.DataFrame: DataFrame with the values set for NaN values
        """
        logger.debug("Set the values for NaN values")

        columns = {
            Columns.COSINE_SIMMILARITY: FillValues.COSINE_SIMMILARITY_XOR,
            Columns.LEVENSTEIHN_DISTANCE: FillValues.LEVENSTEIHN_DISTANCE_XOR,
            Columns.NORMALIZED_LEVENSTEIHN_DISTANCE: FillValues.NORMALIZED_LEVENSTEIHN_DISTANCE_XOR,
            Columns.LEVENSTEIHN_DISTANCE_PROBABILITY: FillValues.LEVENSTEIHN_DISTANCE_PROBABILITY_XOR,
        }

        for column, value in columns.items():
            table.loc[nan_indices, column] = value
        return table

    def _set_metrics_for_and_nan_values(
        self,
        table: pd.DataFrame,
        nan_indices: np.array,
    ) -> pd.DataFrame:
        """
        Set the values for NaN values.

        Args:
            table (pd.DataFrame): DataFrame with the prediction and the ground truth
            nan_indices (np.array): Indices of the rows where either both prediction and ground truth is NaN

        Returns:
            pd.DataFrame: DataFrame with the values set for NaN values
        """
        logger.debug("Set the values for NaN values")

        columns = {
            Columns.COSINE_SIMMILARITY: FillValues.COSINE_SIMMILARITY_AND,
            Columns.LEVENSTEIHN_DISTANCE: FillValues.LEVENSTEIHN_DISTANCE_AND,
            Columns.NORMALIZED_LEVENSTEIHN_DISTANCE: FillValues.NORMALIZED_LEVENSTEIHN_DISTANCE_AND,
            Columns.LEVENSTEIHN_DISTANCE_PROBABILITY: FillValues.LEVENSTEIHN_DISTANCE_PROBABILITY_AND,
        }

        for column, value in columns.items():
            table.loc[nan_indices, column] = value
        return table

    def _get_embeddings_from_api(
        self, values: np.ndarray, url: str
    ) -> np.ndarray:
        """
        Get the embeddings from the API. Sparse embeddings.

        Args:
            values (np.ndarray): Values to get the embeddings for
            url (str): URL of the API

        Returns:
            np.ndarray: Embeddings
        """

        len_values = len(values)
        batched_values = [
            values[i : i + self.embedding_batch_size]
            for i in range(0, len_values, self.embedding_batch_size)
        ]
        embeddings = []
        logger.debug("Get embeddings from the API")
        logger.debug(
            f"Number of batches: {len(batched_values)} of length {int(self.embedding_batch_size)} (subbatches)"
        )
        for batch in batched_values:
            embeddings_batch = get_local_sparse(url, batch)
            embeddings.extend(embeddings_batch)
        return np.array(embeddings)

    def _format_sparse_as_compressed(self, sparse_list: List[List[int]]) -> Any:
        """
        Format sparse embeddings as compressed.

        Args:
            sparse_list (List[List[int]]): Sparse embeddings

        Returns:
            Any: Compressed embeddings
        """
        compressed = {
            Columns.EMBEDDING_INDICES: [],
            Columns.EMBEDDING_VALUES: [],
        }

        for sparse in sparse_list:
            non_zero_indices = np.where(sparse != 0)[0]
            non_zero_values = sparse[non_zero_indices]

            compressed[Columns.EMBEDDING_INDICES].append(
                non_zero_indices.tolist()
            )
            compressed[Columns.EMBEDDING_VALUES].append(
                non_zero_values.tolist()
            )
        return compressed

    def _get_embeddings_batch(
        self, batch: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the embeddings for the batch.

        Args:
            batch (pd.Series): Batch of values (single column)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Embeddings and indices
        """

        batch = batch.to_list()

        logger.debug("Get embeddings for the batch")
        if self.embedding_source == EmbeddingSources.API:
            embeddings = self._get_embeddings_from_api(
                batch, self.embedding_url
            )

        elif self.embedding_source == EmbeddingSources.MODEL:
            embeddings = self.embedding_model.encode(batch)

        return embeddings

    def _calculate_cosine_similarity(self, A: np.array, B: np.array) -> float:
        """
        Calculate the cosine similarity between two matrices.

        NOTE: This function is used only for calculating the cosine similarity
        for vectors that are on the same row in the DataFrame. That is the i-th row in A and i-th column in B.

        Args:
            A (np.array): Matrix A
            B (np.array): Matrix B (not transposed, transposition is done inside the function)

        Returns:
            diagonal_similarities (np.array): Cosine similarities
        """

        logger.debug("Calculate cosine similarity for a batch")
        if not isinstance(A, np.ndarray):
            A = A.to_numpy().reshape(1, -1)
        if not isinstance(B, np.ndarray):
            B = B.to_numpy().reshape(1, -1)
        B = B.T

        # Compute the dot product between rows of A and columns of B
        dot_products = np.einsum("ij,ji->i", A, B)

        # Compute the norms of the rows of A and columns of B
        norm_A = np.linalg.norm(A, axis=1)  # Row-wise norm for A
        norm_B = np.linalg.norm(B, axis=0)  # Column-wise norm for B

        # Compute cosine similarities along the diagonal
        diagonal_similarities = dot_products / (norm_A * norm_B)
        return diagonal_similarities

    def _calculate_levensteihn_distance(
        self, X: pd.Series, y: pd.Series
    ) -> np.ndarray:
        """
        Calculate the Levensteihn distance between two matrices.

        NOTE: This function is used only for calculating the Levensteihn distance
        for vectors that are on the same row in the DataFrame. That is the i-th row in A and i-th column in B.

        Args:
            X (pd.Series): Column X
            y (pd.Series): Column y (ground truth)

        Returns:
            np.ndarray: Levensteihn distances
        """
        logger.debug("Calculate Levensteihn distance for a batch")

        return np.array([Levenshtein.distance(x, y_) for x, y_ in zip(X, y)])

    def _calculate_normalized_levensteihn_distance(
        self, X: pd.Series, y: pd.Series
    ) -> np.ndarray:
        """
        Calculate the normalized Levensteihn distance between two matrices.

        NOTE: This function is used only for calculating the normalized Levensteihn distance
        for vectors that are on the same row in the DataFrame. That is the i-th row in A and i-th column in B.

        Args:
            X (pd.Series): Column X
            y (pd.Series): Column y (ground truth)

        Returns:
            np.ndarray: Normalized Levensteihn distances (Levensteihn distance divided by the length of the ground truth)
        """
        logger.debug("Calculate normalized Levensteihn distance for a batch")
        return np.array(
            [Levenshtein.distance(x, y_) / len(y_) for x, y_ in zip(X, y)]
        )

    def _calculate_levensteihn_distance_probability(
        self, norm_distance: pd.Series
    ) -> np.ndarray:
        """
        Calculate the Levensteihn distance probability.

        Args:
            norm_distance (pd.Series): Levensteihn distances

        Returns:
            np.ndarray: Levensteihn distance probabilities (1 - norm_distance) if norm_distance <1, 0 otherwise
        """
        logger.debug("Calculate Levensteihn distance probability for a batch")
        return np.where(norm_distance < 1, 1 - norm_distance, 0)

    def _process_batch_non_exact_matches(
        self,
        table_batch: pd.DataFrame,
        batch_indices: np.array,
    ) -> pd.DataFrame:
        """
        Process the batch of non-exact matches.

        Args:
            table_batch (pd.DataFrame): DataFrame with the batch of non-exact matches
            batch_indices (np.array): Indices of the batch

        Returns:
            table_batch (pd.DataFrame): DataFrame with the batch of non-exact matches
            embeddings_dict_gt (Dict): Dictionary with the embeddings for the ground truth
            embeddings_dict_pred (Dict): Dictionary with the embeddings for the predictions
        """

        table_batch = table_batch.copy()

        # create embedding dict with batch_indices as keys and {Columns.EMBEDDING_INDICES: None, Columns.EMBEDDING_VALUES: None} as values
        embeddings_dict_pred = {
            index: {
                str(Columns.EMBEDDING_INDICES): None,
                str(Columns.EMBEDDING_VALUES): None,
            }
            for index in batch_indices
        }
        embeddings_dict_gt = embeddings_dict_pred.copy()

        # create metrics dict with metrics as keys and None as values
        metrics_dict = {
            Columns.COSINE_SIMMILARITY: None,
            Columns.LEVENSTEIHN_DISTANCE: None,
            Columns.NORMALIZED_LEVENSTEIHN_DISTANCE: None,
            Columns.LEVENSTEIHN_DISTANCE_PROBABILITY: None,
        }

        logger.debug("Process the batch of non-exact matches")
        # calculate embeddings for the batch
        batch_gt = table_batch.loc[
            batch_indices, Columns.VALUE + Suffixes.GROUND_TRUTH
        ]
        batch_pred = table_batch.loc[
            batch_indices, Columns.VALUE + Suffixes.PREDICTION
        ]

        # get embeddings for the batch and format them as compressed
        embeddings_gt = self._get_embeddings_batch(batch_gt)
        compressed_embeddings_gt = self._format_sparse_as_compressed(
            embeddings_gt
        )

        embeddings_pred = self._get_embeddings_batch(batch_pred)
        compressed_embeddings_pred = self._format_sparse_as_compressed(
            embeddings_pred
        )

        # extract indices and values from the embeddings
        compressed_embeddings_gt_indices = compressed_embeddings_gt.get(
            Columns.EMBEDDING_INDICES
        )
        compressed_embeddings_gt_values = compressed_embeddings_gt.get(
            Columns.EMBEDDING_VALUES
        )
        compressed_embeddings_pred_indices = compressed_embeddings_pred.get(
            Columns.EMBEDDING_INDICES
        )
        compressed_embeddings_pred_values = compressed_embeddings_pred.get(
            Columns.EMBEDDING_VALUES
        )

        # save compressed embeddings to the embeddings_dict
        # TODO: can this be done vectorized?
        for i, index in enumerate(batch_indices):
            embeddings_dict_gt[index][
                str(Columns.EMBEDDING_INDICES)
            ] = compressed_embeddings_gt_indices[i]
            embeddings_dict_gt[index][
                str(Columns.EMBEDDING_VALUES)
            ] = compressed_embeddings_gt_values[i]
            embeddings_dict_pred[index][
                str(Columns.EMBEDDING_INDICES)
            ] = compressed_embeddings_pred_indices[i]
            embeddings_dict_pred[index][
                str(Columns.EMBEDDING_VALUES)
            ] = compressed_embeddings_pred_values[i]

        # calculate ,metrics
        cosine_similarities = self._calculate_cosine_similarity(
            embeddings_gt,
            embeddings_pred,
        )

        levensteihn_distances = self._calculate_levensteihn_distance(
            batch_gt, batch_pred
        )

        normalized_levensteihn_distances = (
            self._calculate_normalized_levensteihn_distance(
                batch_gt, batch_pred
            )
        )

        levensteihn_distance_probabilities = (
            self._calculate_levensteihn_distance_probability(
                normalized_levensteihn_distances
            )
        )

        # set values in the dictionary
        metrics_dict[Columns.COSINE_SIMMILARITY] = cosine_similarities
        metrics_dict[Columns.LEVENSTEIHN_DISTANCE] = levensteihn_distances
        metrics_dict[
            Columns.NORMALIZED_LEVENSTEIHN_DISTANCE
        ] = normalized_levensteihn_distances
        metrics_dict[
            Columns.LEVENSTEIHN_DISTANCE_PROBABILITY
        ] = levensteihn_distance_probabilities

        return metrics_dict, embeddings_dict_gt, embeddings_dict_pred

    def _get_raw_predictions(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get the raw predictions.

        Args:
            X (pd.DataFrame): DataFrame with the predictions
            y (pd.DataFrame): DataFrame with the ground truth

        Returns:
            pd.DataFrame: DataFrame with the predictions and the ground truth
            total_embedding_dict_gt (Dict): Dictionary with the embeddings for the ground truth
            total_embedding_dict_pred (Dict): Dictionary with the embeddings for the predictions
        """

        logger.debug("Get the raw predictions")
        total_embedding_dict_gt = {
            i: {
                str(Columns.EMBEDDING_INDICES): None,
                str(Columns.EMBEDDING_VALUES): None,
            }
            for i in range(len(X))
        }

        total_embedding_dict_pred = total_embedding_dict_gt.copy()

        logger.debug("Merge the DataFrames")
        table = self._merge_x_y(X, y)

        table = self._preprocess_merged_data(table)

        # calculate exact match
        exact_match_indices = self._calculate_exact_match(
            table[Columns.VALUE + Suffixes.PREDICTION],
            table[Columns.VALUE + Suffixes.GROUND_TRUTH],
            include_nan=True,
        )

        # calculate non-exact match indices
        non_exact_match_indices = self._calculate_non_exact_match_indices(
            table[Columns.VALUE + Suffixes.PREDICTION],
            table[Columns.VALUE + Suffixes.GROUND_TRUTH],
        )

        # calculate nan indices
        nan_indices_xor = self._calculate_xor_nan_indices(
            table[Columns.VALUE + Suffixes.PREDICTION],
            table[Columns.VALUE + Suffixes.GROUND_TRUTH],
        )
        nan_indices_and = self._calculate_and_nan_indices(
            table[Columns.VALUE + Suffixes.PREDICTION],
            table[Columns.VALUE + Suffixes.GROUND_TRUTH],
        )

        # for each exact match, set cosine similarity to 1, levensteihn distance to 0, normalized levensteihn distance to 0 and levensteihn distance probability to 1
        table = self._set_metrics_for_exact_matches(table, exact_match_indices)

        # set values for xor nan values
        table = self._set_metrics_for_xor_nan_values(table, nan_indices_xor)

        # remove nan indices from non-exact match indices
        non_exact_match_indices = list(
            set(non_exact_match_indices)
            .difference(set(nan_indices_xor))
            .difference(set(nan_indices_and))
        )

        # for each non-exact match, calculate cosine similarity, levensteihn distance, normalized levensteihn distance and levensteihn distance probability
        logger.debug("Calculate the values for non-exact matches")
        logger.debug(
            f"Number of non-exact matches: {len(non_exact_match_indices)}"
        )

        # calculate values for non-exact matches batch by batch
        batches = [
            non_exact_match_indices[i : i + self.batch_size]
            for i in range(0, len(non_exact_match_indices), self.batch_size)
        ]
        logger.debug(
            f"Number of batches: {len(batches)} of length {int(self.batch_size)}"
        )

        for i, batch_indices in tqdm(
            enumerate(batches),
            desc="Calculating values for non-exact matches",
            total=len(batches),
        ):
            batch_table = table.loc[batch_indices]

            (
                metrics_dict,
                embeddings_dict_gt,
                embeddings_dict_pred,
            ) = self._process_batch_non_exact_matches(
                batch_table,
                batch_indices,
            )

            # set metrics
            for metric, values in metrics_dict.items():
                table.loc[batch_indices, metric] = values

            # set embeddings
            total_embedding_dict_gt.update(embeddings_dict_gt)
            total_embedding_dict_pred.update(embeddings_dict_pred)

        return table, total_embedding_dict_gt, total_embedding_dict_pred

    def predict_proba(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        mode="cosine",
        set_training_report=False,
    ) -> np.ndarray:
        """
        Predict the probabilities of the predictions.

        Args:
            X (pd.DataFrame): DataFrame with the predictions
            y (pd.DataFrame): DataFrame with the ground truth
            mode (str): Mode of the prediction (predict or fit)
            set_training_report (bool): Set the training report

        Returns:
            np.ndarray: Probabilities (if mode is 'predict')
            OR
            pd.DataFrame: DataFrame with the probabilities and the metrics (if mode is 'fit')
        """

        logger.info(f"predict_proba: {mode = }")

        return_dict = {
            PredictProbaModes.LEVENSTEIHN: Columns.LEVENSTEIHN_DISTANCE_PROBABILITY,
            PredictProbaModes.COSINE: Columns.COSINE_SIMMILARITY,
            PredictProbaModes.FIT: [
                Columns.INSTANCE_ID,
                Columns.SUB_INSTANCE_ID,
                Columns.ENTITY,
                Columns.VALUE + Suffixes.GROUND_TRUTH,
                Columns.VALUE + Suffixes.PREDICTION,
                Columns.COSINE_SIMMILARITY,
                Columns.LEVENSTEIHN_DISTANCE,
                Columns.NORMALIZED_LEVENSTEIHN_DISTANCE,
                Columns.LEVENSTEIHN_DISTANCE_PROBABILITY,
            ],
        }

        assert mode in [
            PredictProbaModes.LEVENSTEIHN,
            PredictProbaModes.COSINE,
            PredictProbaModes.FIT,
        ], f"Mode must be either {PredictProbaModes.LEVENSTEIHN}, {PredictProbaModes.COSINE} or {PredictProbaModes.FIT}"

        logger.debug(f"Predicting probabilities using mode {mode}")
        (
            table,
            total_embedding_dict_gt,
            total_embedding_dict_pred,
        ) = self._get_raw_predictions(X, y)

        print(table)
        print(return_dict)
        print(mode)

        return_values = table.get(return_dict[mode])

        if set_training_report:
            self.training_report = table
            self.embedding_dict_gt = total_embedding_dict_gt
            self.embedding_dict_pred = total_embedding_dict_pred

        if return_values is not None:
            return np.array(return_values)
        else:
            raise ValueError(f"Return values for mode {mode} not found")

    def _calculate_labels(
        self, probabilities: Union[pd.Series, np.array]
    ) -> List[float]:
        """
        Calculate the labels.

        Args:
            probabilities (Union[pd.Series, np.array]): Probabilities

        Returns:
            float: labels
        """
        if not isinstance(probabilities, np.ndarray):
            probabilities = probabilities.to_numpy()

        logger.debug(
            f"Calculate labels using threshold {float(self.threshold)}"
        )
        return np.where(probabilities >= self.threshold, 1, 0)

    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        mode: str = "cosine",
    ) -> np.ndarray:
        """
        Predict the labels.

        Args:
            X (Union[pd.DataFrame, pd.Series]): DataFrame with the predictions
            y (Union[pd.DataFrame, pd.Series]): DataFrame with the ground truth
            mode (str): Mode of the prediction

        Returns:
            np.ndarray: Labels
        """
        logger.debug(
            f"Predicting ------------------------------------------------------"
        )

        predictions = self.predict_proba(X, y, mode=mode)

        labels = self._calculate_labels(predictions)  # TODO zmianic na labels

        logger.debug(
            "Finished predicting ----------------------------------------------"
        )

        return labels

    def _calculate_heuristic_threshold(
        self,
        probabilities: Union[pd.Series, np.array],
        coefficient: float = DefaultFloats.THRESHOLD_COEFFICIENT,
    ) -> float:
        """
        Calculate the heuristic threshold. Uses the IQR.
        Formula: Q1 - coefficient * IQR

        Args:
            probabilities (Union[pd.Series, np.array]): Probabilities
            coefficient (float): Coefficient for the heuristic threshold using IQR (default is 1.5)

        Returns:
            float: Heuristic threshold
        """
        logger.debug("Calculate heuristic threshold")
        iqr = np.percentile(probabilities, 75) - np.percentile(
            probabilities, 25
        )
        threshold = np.percentile(probabilities, 25) - coefficient * iqr
        logger.debug(f"Heurisitc threshold: {threshold}")
        return threshold

    def _set_threshold(
        self,
        probabilities: Union[pd.Series, np.array],
        coefficient: float = DefaultFloats.THRESHOLD_COEFFICIENT,
    ):
        """
        Set the threshold.

        Args:
            probabilities (Union[pd.Series, np.array]): Probabilities
            coefficient (float): Coefficient for the heuristic threshold using IQR (default is 1.5)
        """

        logger.debug("Setting the threshold")

        self.threshold = self._calculate_heuristic_threshold(
            probabilities=probabilities,
            coefficient=coefficient,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y: Union[pd.DataFrame, pd.Series],
        coefficient: float = 0.5,
        mode: str = "cosine",
    ) -> Union[float, Dict[str, float]]:
        """
        Fit the model.

        Args:
            X (Union[pd.DataFrame, pd.Series]): DataFrame with the predictions
            y (Union[pd.DataFrame, pd.Series]): DataFrame with the ground truth
            coefficient (float): Coefficient for the heuristic threshold using IQR (default is 1.5)
            mode (str): Mode of the prediction
        """

        logger.debug(
            f"Fitting the model------------------------------------------------------"
        )
        self.X = X
        self.y = y

        # FIT mode is used, as .fit() should set training_report
        probas = self.predict_proba(X, y, mode=mode, set_training_report=True)

        # this could be done in
        self._set_threshold(
            probas,
            coefficient=coefficient,
        )

        logger.debug(
            "Finished fitting the model----------------------------------------------"
        )
