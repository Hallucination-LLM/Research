import os
import re
import logging
import Levenshtein
import numpy as np
import pandas as pd
from golemai.enums import OcrEvaluationResultsKeys, StatsKeys

logger = logging.getLogger(__name__)


class OcrEvaluator():
    """    
    OcrEvaluator is a class for evaluating Optical Character Recognition (OCR) results against ground truth data.
    Attributes:
        ocr_dir_path (str): Path to the directory containing OCR results.
        ground_true_dir_path (str): Path to the directory containing ground truth data.
        template_words (set): A set of sanitized and lowercased words from the template file.
    Public Methods:
        calculate_ocr_metrics(doc_types):
            Calculates OCR metrics for the given document types.
        calculate_ocr_stats(doc_types):
            Calculates OCR statistics for given document types, computing various statistical metrics for OCR results.
    Usage example:
        >>> evaluator = OcrEvaluator("data/ocr_dir", "data/ground_truth_dir", "data/template_file.txt")
        >>> evaluator.calculate_ocr_metrics(["handwritten"])
        >>> evaluator.calculate_ocr_stats(["handwritten"])
    """

    def __init__(self, ocr_dir_path, ground_true_dir_path, template_file_path):
        """
        Initializes the OCREvaluator

        Args:
            ocr_dir_path (str): Path to the directory containing OCR results.
            ground_true_dir_path (str): Path to the directory containing ground truth data.
            template_file_path (str): Path to the file containing template words.
        """
        self.ocr_dir_path = ocr_dir_path
        self.ground_true_dir_path = ground_true_dir_path
        self.template_words = self._load_template_words(template_file_path)

    def _load_template_words(self, template_file_path):
        """
        Loads and processes template words from a specified file.
        Args:
            template_file_path (str): The path to the template file containing words.
        Returns:
            set: A list of sanitized and lowercased words from the template file.
        """
        logger.debug(f"load_template_words: {template_file_path = }")
        with open(template_file_path, 'r', encoding='utf-8') as f:
            template_data = self._sanitize_text(" ".join([line.strip() for line in f.readlines()]))
        return template_data.lower().split(" ")

    def _sanitize_text(self, text):
        """
        Sanitizes the input text by performing the following operations:
        1. Replaces multiple whitespace characters with a single space and trims leading/trailing whitespace.
        2. Removes empty square brackets and digits followed by a closing parenthesis.
        3. Replaces occurrences of '- -' with a single '-'.
        Args:
            text (str): The input text to be sanitized.
        Returns:
            str: The sanitized text.
        """
        logger.debug(f"sanitize_text: len(text) = {len(text)}")
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[\s*\]|\d+\)', '', text)
        text = re.sub(r'- -', '-', text)
        return text

    def _filter_numbers(self, text_data):
        """
        Filters out alphabetic characters from words in the given text data that contain numbers.
        Args:
            text_data (list of str): A list of words to be filtered.
        Returns:
            list of str: A list of words with alphabetic characters removed, containing only numbers and other non-alphabetic characters.
        """
        logger.debug(f"filter_numbers: len(text_data) = {len(text_data)}")
        return [re.sub(r'[a-zA-Z]', '', w) for w in text_data if re.search(r'\d', w)]

    def _filter_content(self, text_data):
        """
        Filters and processes the input text data.
        This method performs the following operations on the input text data:
        1. Converts all text to lowercase.
        2. Splits the text into individual words.
        3. Strips leading and trailing quotation marks (both single and double) from each word.
        4. Excludes words that are present in the `template_words` attribute.
        5. Excludes single-character words unless they are alphanumeric.
        Args:
            text_data (str): The input text data to be filtered and processed.
        Returns:
            list: A list of filtered and processed words.
        """
        logger.debug(f"filter_content: len(text_data) = {len(text_data)}")
        return [
            word.strip('"\'')
            for word in text_data.lower().split()
            if (cleaned_word := word.strip('"\''))
            and cleaned_word not in self.template_words
            and (len(cleaned_word) != 1 or cleaned_word.isalnum())
        ]

    def _calculate_ratio(self, achieved_metric, gt_data):
        """
        Calculate the ratio of the achieved metric to the ground truth data length.
        This method computes the ratio by subtracting the achieved metric divided by 
        the length of the ground truth data from 1. If the length of the ground truth 
        data is zero, it returns 1.
        Args:
            achieved_metric (float): The metric that has been achieved.
            gt_data (list): The ground truth data.
        Returns:
            float: The calculated ratio.
        """
        logger.debug(f"calculate_ratio: {achieved_metric = }, len(gt_data) = {len(gt_data)}")
        return 1 - achieved_metric / len(gt_data) if len(gt_data) > 0 else 1

    def _extract_text_from_file(self, file_path):
        """
        Extracts and sanitizes text from a given file.
        Args:
            file_path (str): The path to the file from which to extract text.
        Returns:
            str: The sanitized text extracted from the file.
        """
        logger.debug(f"extract_text_from_file: {file_path = }")
        with open(file_path, 'r', encoding='utf-8') as file:
            return self._sanitize_text(" ".join([line.strip() for line in file.readlines()]))

    def _merge_ocr_texts(self, ocr_texts):
        """
        Merges a list of OCR text segments into a list of combined text segments.
        This method processes a list of OCR text segments, combining consecutive single-character
        segments into a single string and appending multi-character segments directly to the result list.
        Args:
            ocr_texts (list of str): A list of OCR text segments to be merged.
        Returns:
            list of str: A list of merged OCR text segments.
        """
        logger.debug(f"merge_ocr_texts: len(ocr_texts) = {len(ocr_texts)}")
        merged_ocr = []
        temp_buffer = []
        for text in ocr_texts:
            if len(text) == 1:
                temp_buffer.append(text)
            else:
                if temp_buffer:
                    merged_ocr.append("".join(temp_buffer))
                    temp_buffer = []
                merged_ocr.append(text)
        if temp_buffer:
            merged_ocr.append("".join(temp_buffer))
        return merged_ocr

    def _calculate_ocr_discrepancy_windowed(self, ground_truth, ocr_predictions, window_size=3):
        """
        Calculate the OCR discrepancy between the ground truth and OCR predictions.

        This method compares the OCR predictions with the ground truth within a specified window size
        and calculates the Levenshtein distance between the unmatched elements in the ground truth and
        the OCR predictions that are not present in the ground truth.

        Args:
            ground_truth (list): The list of ground truth elements.
            ocr_predictions (list): The list of OCR predicted elements.
            window_size (int, optional): The size of the window to search for matching elements. Defaults to 3.

        Returns:
            tuple: A tuple containing:
                - int: The Levenshtein distance between the unmatched elements in the ground truth and the OCR predictions.
                - int: The length of the ground truth elements that were not matched.
        """
        logger.debug(f"calculate_ocr_discrepancy: len(ground_truth) = {len(ground_truth)}, len(ocr_predictions) = {len(ocr_predictions)}, {window_size = }")
        already_found_flag = '<ALREADY_FOUND>'
        ground_truth_copy = ground_truth.copy()
        not_present_in_gt = []
        for i, ocr_element in enumerate(ocr_predictions):
            start_index = min(max(0, i - window_size - 2), len(ground_truth_copy) - window_size)
            end_index = min(len(ground_truth_copy), i + window_size - 1)
            window_elements = ground_truth_copy[start_index:end_index]
            if ocr_element in window_elements:
                to_exclude_idx = start_index + window_elements.index(ocr_element)
                ground_truth_copy[to_exclude_idx] = already_found_flag
                continue
            for j, window_element in enumerate(window_elements):
                if ocr_element in window_element:
                    rest = window_element.replace(ocr_element, "")
                    to_exclude_idx = start_index + j
                    ground_truth_copy[to_exclude_idx] = rest
                    break
            else:
                not_present_in_gt.append(ocr_element)
        ground_truth_copy = [e for e in ground_truth_copy if e != already_found_flag]
        return Levenshtein.distance("".join(ground_truth_copy), "".join(not_present_in_gt)), len(ground_truth_copy)

    def calculate_ocr_metrics(self, doc_types):
        """
        Calculate OCR metrics for the given document types.
        Args:
            doc_types (list): List of document types to evaluate.
        Returns:
            dict: A dictionary containing OCR evaluation results for each file. The keys are file names and the values are dictionaries with the following keys:
            - OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_ALL_TEXT: Ratio of correct characters in the entire text.
            - OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_CONTENT: Ratio of correct characters in the filtered content.
            - OcrEvaluationResultsKeys.CORRECT_WORDS_RATIO_CONTENT: Ratio of correct words in the filtered content.
            - OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_NUMS: Ratio of correct characters in the numbers.
            - OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_CUSTOM_NUMS: Custom ratio of correct characters in the numbers.
            - OcrEvaluationResultsKeys.CORRECT_WORDS_RATIO_CUSTOM_NUMS: Custom ratio of correct words in the numbers.
        """
        logger.info(f"calculate_ocr_metrics: {doc_types = }")
        ocr_results = {}
        for doc_type in doc_types:

            doc_type_path = os.path.join(self.ocr_dir_path, doc_type)
            for file_name in os.listdir(doc_type_path):

                logging.info(f"Processing file: {file_name}")

                ocr_file_path = os.path.join(doc_type_path, file_name)
                ground_true_path = os.path.join(self.ground_true_dir_path, file_name)

                logger.debug(f"OCR file path: {ocr_file_path}")
                logger.debug(f"Ground truth file path: {ground_true_path}")

                ocr_data = self._extract_text_from_file(ocr_file_path) 
                ground_true_data = self._extract_text_from_file(ground_true_path)

                ground_true_only_content = self._filter_content(ground_true_data)
                ocr_only_content = self._filter_content(ocr_data)

                only_numbers_gt = self._filter_numbers(ground_true_only_content)
                only_numbers_ocr = self._filter_numbers(ocr_only_content)

                joined_only_content_gt = "".join(ground_true_only_content)
                joined_only_numbers_gt = "".join(only_numbers_gt)

                only_numbers_ocr = self._merge_ocr_texts(only_numbers_ocr)

                lavensthein_my_aproach, not_detected_words_num = self._calculate_ocr_discrepancy_windowed(only_numbers_gt, only_numbers_ocr)

                lavenshtein_distance_all_text = Levenshtein.distance(ocr_data, ground_true_data)
                lavenshtein_distance_content = Levenshtein.distance("".join(ocr_only_content), "".join(ground_true_only_content))
                lavenshtein_distance_nums = Levenshtein.distance(joined_only_numbers_gt, "".join(only_numbers_ocr))
                correct_words_content = [w in ground_true_only_content for w in ocr_only_content]

                logger.debug(f"File: {file_name}, Levenshtein distance (all text): {lavenshtein_distance_all_text}")
                logger.debug(f"File: {file_name}, Levenshtein distance (content): {lavenshtein_distance_content}")
                logger.debug(f"File: {file_name}, Levenshtein distance (numbers): {lavenshtein_distance_nums}")

                ocr_results[file_name] = {
                    OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_ALL_TEXT: self._calculate_ratio(lavenshtein_distance_all_text, ground_true_data),
                    OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_CONTENT: self._calculate_ratio(lavenshtein_distance_content, joined_only_content_gt),
                    OcrEvaluationResultsKeys.CORRECT_WORDS_RATIO_CONTENT: np.mean(correct_words_content),
                    OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_NUMS: self._calculate_ratio(lavenshtein_distance_nums, joined_only_numbers_gt),
                    OcrEvaluationResultsKeys.CORRECT_CHARS_RATIO_CUSTOM_NUMS: self._calculate_ratio(lavensthein_my_aproach, joined_only_numbers_gt),
                    OcrEvaluationResultsKeys.CORRECT_WORDS_RATIO_CUSTOM_NUMS: self._calculate_ratio(not_detected_words_num, only_numbers_gt)
                }
                logger.debug(f"OCR metrics for file {file_name}:\n" + "\n".join([f"{key}: {value}" for key, value in ocr_results[file_name].items()]))

        logger.info("Completed OCR metric calculation.")
        return ocr_results

    def calculate_ocr_stats(self, doc_types):
        """
        Calculate OCR statistics for given document types.
        This method computes various statistical metrics (mean, median, standard deviation, minimum, and maximum)
        for OCR results across different document types. The statistics are calculated for each metric returned
        by the `calculate_ocr_metrics` method.
        Args:
            doc_types (list): A list of document types to evaluate OCR metrics for.
        Returns:
            pd.DataFrame: A DataFrame containing the calculated statistics for each metric, rounded to three decimal places.
                  The DataFrame is transposed such that each row represents a metric and each column represents a statistic.
        """
        logger.info(f"calculate_ocr_stats: {doc_types = }")
        ocr_results = self.calculate_ocr_metrics(doc_types)
        mean_metrics = {metric: [] for metric in next(iter(ocr_results.values())).keys()}

        for metrics in ocr_results.values():
            for metric, value in metrics.items():
                mean_metrics[metric].append(value)

        stats = {
            metric: {
                StatsKeys.MEAN: np.mean(values),
                StatsKeys.MEDIAN: np.median(values),
                StatsKeys.STD: np.std(values),
                StatsKeys.MIN: np.min(values),
                StatsKeys.MAX: np.max(values)
            }
            for metric, values in mean_metrics.items()
        }

        stats_df = pd.DataFrame(stats).round(3).T
        logger.info("Completed OCR stats calculation.")
        logger.debug(f"Final stats DataFrame:\n{stats_df}")
        return stats_df
