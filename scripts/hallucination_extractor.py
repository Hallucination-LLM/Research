from typing import List, Tuple
from golemai.nlp.llm_resp_gen import LLMRespGen
from .extractor_enums import DfColumnsEnum, EvalDfColumnsEnum
import pandas as pd


class HallucinationExtractor:
    def __init__(self, eval_path: str, dataset_path: str, output_path: str, llm_rg: LLMRespGen):
        self.output_path = output_path
        self.eval_df = pd.read_json(eval_path, lines=True)
        self.df = pd.read_parquet(dataset_path).reset_index()

        self._llm_rg = llm_rg

    def process_data(self):
        self._keep_valid_indices()
        self._set_from_eval()
        self._remove_from_spans()
        self._remove_errors()

        self._handle_indices()
        self._rename_columns()
        self._calculate_hallu_features()

    def _keep_valid_indices(self):
        self.df = self.df.loc[self.df.index.isin(self.eval_df.index)]

    def _set_from_eval(self):
        self.df[DfColumnsEnum.PROBLEMATIC] = self.eval_df[EvalDfColumnsEnum.PROBLEMATIC].values
        self.df[DfColumnsEnum.RESPONSE] = self.eval_df[EvalDfColumnsEnum.RESPONSE].values

    def _remove_from_spans(self):
        self.df[DfColumnsEnum.PROBLEMATIC] = self.df[DfColumnsEnum.PROBLEMATIC].apply(lambda x: [span.removeprefix('\"').removesuffix('\"') for span in x] if x is not None else None)

    def _remove_errors(self):
        self.df = self.df.loc[self.df[DfColumnsEnum.RESPONSE] != '<CUDA_ERROR>']

    def _handle_indices(self):
        self.df[DfColumnsEnum.GPT_INDEX] = self.df.index
        self.df['indices'] = self.df.apply(
            lambda row: self._find_indices(row[DfColumnsEnum.RESPONSE], row[DfColumnsEnum.PROBLEMATIC]), axis=1)
        self.df = self.df.loc[~self.df[DfColumnsEnum.IDX].isin(
            [row[DfColumnsEnum.IDX] for i, row in self.df.iterrows() if
             any(index == (None, None) for index in row['indices'])])]
        self.df = self.df.drop(columns=['indices'])

    def _rename_columns(self):
        self.df = self.df.rename(columns={'question': 'query'})

    def _calculate_hallu_features(self):
        self.df[[DfColumnsEnum.PROMPT_LEN, DfColumnsEnum.CONTEXT]] = self.df.apply(self._format_context, axis=1)
        self.df[DfColumnsEnum.HALLU_IDX] = self.df.apply(lambda row: self._find_indices(row[DfColumnsEnum.CONTEXT],
                                                                                        row[DfColumnsEnum.PROBLEMATIC]),
                                                         axis=1)
        self.df[DfColumnsEnum.HALLU_TOKENS] = self.df.apply(
            lambda row: self._find_hallu_tokens(row[DfColumnsEnum.CONTEXT],
                                                row[DfColumnsEnum.HALLU_IDX]), axis=1)
        self.df[DfColumnsEnum.CONTAIN_HALLU] = self.df[DfColumnsEnum.PROBLEMATIC].apply(lambda x: True if x else False)

    @staticmethod
    def _find_indices(response: str, spans: List[str] | str) -> List[Tuple[int, int]]:
        """Find indices of each substring in 'problematic_spans' within 'model_response'."""
        indices = []

        if isinstance(spans, str):
            spans = [spans]
        elif spans is None:
            return

        for span in spans:
            start = response.rfind(span)
            if start != -1:
                end = start + len(span)
                indices.append((start, end))
            else:
                indices.append((None, None))

        return indices

    def _format_context(self, row):
        formatted_prompt = self._llm_rg._get_ready_prompt(row=row,
                                                          prompt_columns=['query', 'context'])
        prompt_length = self._llm_rg.tokenizer(formatted_prompt,
                                               return_tensors="pt",
                                               padding=True)['inputs_ids'].shape[1]
        formatted_context = f"{formatted_prompt}{row[DfColumnsEnum.RESPONSE]}"
        return prompt_length, formatted_context

    def _find_hallu_tokens(self, formatted_context: str, hallu_indices: List[Tuple[int, int]]) -> List[bool]:
        offset_mapping = self._llm_rg.tokenizer(formatted_context,
                                                return_tensors="pt",
                                                padding=True,
                                                return_offsets_mapping=True)['offset_mapping'][0].tolist()
        hallu_tokens = [False] * len(offset_mapping)

        for idx, (start, end) in enumerate(offset_mapping):
            for hallu_start, hallu_end in hallu_indices:
                if start >= hallu_start and end <= hallu_end:
                    hallu_tokens[idx] = True

        return hallu_tokens

    def save_data(self):
        self.df.to_parquet(self.output_path)

    def run(self):
        self.process_data()
        self.save_data()
