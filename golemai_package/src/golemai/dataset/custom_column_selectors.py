from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
from typing import Tuple
import logging
from golemai.config import LOGGER_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

class JensenShannonSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int = 10) -> None:
        """
        Initialize the JensenShannonSelector.

        Args:
            n_features (int): The number of features to select.
        """
        self.n_features = n_features
        self.selected_features_: np.ndarray | None = None

    def get_prob_vec(self, df: pd.DataFrame, col: str, target_col: str, min_val: int, max_val: int) -> Tuple[pd.Series, pd.Series]:
        """
        Get the probability vectors for positive and negative classes.

        Args:
            df (pd.DataFrame): The input DataFrame.
            col (str): The column for which to calculate probabilities.
            target_col (str): The target column indicating class labels.
            min_val (int, optional): The minimum value for the range.
            max_val (int, optional): The maximum value for the range.

        Returns:
            Tuple[pd.Series, pd.Series]: The probability vectors for positive and negative classes.
        """

        logger.debug(f'get_prob_vec: {col = }, {target_col = }, {min_val = }, {max_val = }')

        all_values = np.arange(min_val, max_val + 1)
        pos_val, neg_val = df[target_col].value_counts().index
        pos = df.loc[df[target_col] == pos_val, col].value_counts(normalize=True).reindex(all_values, fill_value=0).sort_index()
        neg = df.loc[df[target_col] == neg_val, col].value_counts(normalize=True).reindex(all_values, fill_value=0).sort_index()
        return pos, neg

    def calculate_probabilities(self, df: pd.DataFrame, label_col: str, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate positive and negative probability vectors for each column in the dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing numerical data.
            label_col (str): The name of the column containing the labels.
            n_bins (int): The number of bins to use for probability calculation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing the positive and negative probabilities.
        """

        logger.debug(f'calculate_probabilities: {label_col = }, {n_bins = }')

        process_cols = [col for col in df.columns if col not in [label_col, 'dataset']]
        
        pos_probs = np.zeros((len(process_cols), n_bins))
        neg_probs = np.zeros((len(process_cols), n_bins))

        for i, col in enumerate(process_cols):
            pos, neg = self.get_prob_vec(df, col, label_col, max_val=n_bins - 1, min_val=0)
            pos_probs[i] = pos.values
            neg_probs[i] = neg.values

        return pos_probs, neg_probs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'JensenShannonSelector':
        """
        Fit the selector to the data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
        Returns:
            JensenShannonSelector: The fitted selector.
        """

        logger.info(f'fit: {X.shape = }, {y.shape = }')

        X.dropna(inplace=True)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
        numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.to_list()

        n_bins = int(2 * (len(X) ** (1/3)))
        binned_df = X[numerical_cols].apply(pd.qcut, q=n_bins, labels=False, duplicates='drop')
        binned_df = pd.concat([binned_df, X[categorical_cols]], axis=1)
        binned_df['label'] = y.values

        pos_probs, neg_probs = self.calculate_probabilities(binned_df, 'label', n_bins)
        js_divs = jensenshannon(pos_probs, neg_probs, axis=1)

        print(categorical_cols)

        jensen_divs_df = pd.DataFrame(
            js_divs, 
            index=numerical_cols + categorical_cols, 
            columns=['js_div']
        )

        self.selected_features_ = jensen_divs_df.nlargest(self.n_features, 'js_div').index.to_list()
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The selected features.
        """

        logger.info(f'transform: {X.shape = }')

        return X[self.selected_features_]
    


class ProportionAggSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int = 10, agg_func: str = 'median', keep_categorical: bool = True) -> None:
        """
        Initialize the ProportionAggSelector.

        Args:
            n_features (int): The number of features to select.
            agg_func (str): The aggregation function to use (default is 'median').
            keep_categorical (bool): Whether to keep categorical columns (default is True).
        """
        self.n_features = n_features
        self.agg_func = agg_func
        self.keep_categorical = keep_categorical
        self.selected_features_: list | None = None

    def _get_grouped_stats(
        self, X: pd.DataFrame, numerical_cols: list[str]
    ) -> pd.DataFrame:
        
        """
        Get the grouped statistics for the numerical columns.

        Args:
            X (pd.DataFrame): The input DataFrame.
            numerical_cols (list[str]): The numerical columns.

        Returns:
            pd.DataFrame: The grouped statistics.
        """
        
        stats_grouped = X[numerical_cols].groupby('label').agg([self.agg_func]).T
        stats_grouped['proportion'] = stats_grouped[0] / stats_grouped[1]
        stats_grouped = stats_grouped.reset_index().rename(columns={'level_0': 'feature', 'level_1': 'statistic'})

        return stats_grouped

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProportionAggSelector':
        """
        Fit the selector to the data.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.

        Returns:
            ProportionAggSelector: The fitted selector.
        """

        logger.info(f'fit: {X.shape = }, {y.shape = }')

        X['label'] = y.values

        numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.to_list()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()

        stats_grouped = self._get_grouped_stats(X, numerical_cols)

        top_features = stats_grouped.nlargest(self.n_features // 2, 'proportion')['feature'].values.tolist()
        bottom_features = stats_grouped.nsmallest(self.n_features // 2, 'proportion')['feature'].values.tolist()
        
        self.selected_features_ = top_features + bottom_features
        
        if self.keep_categorical:
            self.selected_features_ += categorical_cols
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The selected features.
        """
        return X[self.selected_features_]