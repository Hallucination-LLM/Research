import logging
from ast import literal_eval
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from golemai.config import LOGGER_LEVEL
from golemai.enums import LocalizationPredictorColumns as Columns
from golemai.logging.log_formater import init_logger

# Create logger
logger = init_logger(LOGGER_LEVEL)


class NerDictParser:
    def __init__(
        self,
        id_column: str = "instance_ID",
        sub_id_column: str = "sub_instance_ID",
    ):
        self.id_column = id_column
        self.sub_id_column = sub_id_column

    def _convert_from_nested_dict_to_flat_dataframe_row_level(
        self, data: Dict[str, Any], prefix: str = ""
    ) -> pd.DataFrame:
        """
        Converts a dictionary to a flat DataFrame.
        So there must be a ground_truth and prediction column in the DataFrame.
        Both columns contain dictionaries.
        They should be unnested and the result should be a DataFrame.
        The GT and Pred columns should have a prefix, so that the columns are not overwritten.

        There is one level of nesting in the dictionaries.

        Args:
            data (Dict[str, Any]):  dictionary
            prefix (str): Prefix for the columns. Default is ""

        Returns:
            pd.DataFrame: Flat DataFrame
        """

        logger.debug(f"Converting nested dictionary to flat DataFrame")
        flat_data = pd.DataFrame()

        if not isinstance(data, dict):
            data = literal_eval(data)

        for key, value in data.items():
            flat_data[prefix + key] = [value]

        return flat_data

    def _preprocess_data(
        self, data: Union[pd.DataFrame, pd.Series], dict_column: str
    ) -> pd.Series:
        """
        Preprocesses the input data. The rows of the DataFrame are dictionaries.
        The output is a DataFrame with the dictionaries unnested.

        Args:
            data (Union[pd.DataFrame, pd.Series]): Input data
            dict_column (str): Name of the column containing the dictionaries

        Returns:
            pd.DataFrame: Preprocessed data
        """

        # add ID column
        data = data.copy()

        flattened_data = pd.DataFrame()
        flattened_data[self.id_column] = None

        for index in range(data.shape[0]):
            to_add = self._convert_from_nested_dict_to_flat_dataframe_row_level(
                data.loc[index, dict_column]
            )
            to_add = to_add.melt()
            to_add[self.id_column] = index

            flattened_data = pd.concat(
                [flattened_data, to_add], axis=0, ignore_index=True
            )

        # make ID the first column]
        flattened_data = flattened_data[
            [self.id_column]
            + [col for col in flattened_data.columns if col != self.id_column]
        ]

        flattened_data[self.sub_id_column] = 1

        for index in range(flattened_data.shape[0]):
            if isinstance(flattened_data.iloc[index][Columns.VALUE], list):
                to_add = pd.DataFrame(
                    {
                        self.id_column: [
                            flattened_data.iloc[index][self.id_column]
                        ]
                        * len(flattened_data.iloc[index][Columns.VALUE]),
                        self.sub_id_column: range(
                            1,
                            len(flattened_data.iloc[index][Columns.VALUE]) + 1,
                        ),
                        Columns.VARIABLE: flattened_data.iloc[index][
                            Columns.VARIABLE
                        ],
                        Columns.VALUE: None,
                    }
                )

                for i, value in enumerate(
                    flattened_data.iloc[index][Columns.VALUE]
                ):
                    to_add.loc[i, Columns.VALUE] = value

                flattened_data = pd.concat(
                    [flattened_data, to_add], axis=0, ignore_index=True
                )
                # remove the original row
                flattened_data.drop(index, inplace=True)

        # convert empty strings to None
        flattened_data.replace(Columns.EMPTY_STRING, np.nan, inplace=True)

        # raname variable to entity
        flattened_data.rename(
            columns={Columns.VARIABLE: Columns.ENTITY}, inplace=True
        )

        # move self.sub_id_column to the second column
        flattened_data = flattened_data[
            [self.id_column, self.sub_id_column, Columns.ENTITY, Columns.VALUE]
        ]

        # order by instance ID then by sub instance ID
        flattened_data.sort_values(
            by=[self.id_column, self.sub_id_column], inplace=True
        )
        flattened_data.reset_index(drop=True, inplace=True)

        return flattened_data

    def parse(self, data: pd.DataFrame, dict_column="entities") -> pd.DataFrame:
        """
        Parses the input data

        Args:
            data (pd.DataFrame): Input data
            dict_column (str): Name of the column containing the dictionaries

        Returns:
            pd.DataFrame: Parsed data
        """
        logger.info(
            "parse: Parsing the input data to the format for the NER model"
        )
        data = self._preprocess_data(data, dict_column)
        return data
