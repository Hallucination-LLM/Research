from typing import List, Tuple
import os
import numpy as np
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from golemai.nlp.llm_resp_gen import LLMRespGen

class HallucinationDatasetExtractor:
    def __init__(self, df: pd.DataFrame, llm_rg: LLMRespGen, att_dir_path: str = 'attensions'):
        self.df = df
        self._llm_rg = llm_rg
        self._setup()

        self.att_dir_path = att_dir_path
        self.context_span = '`CONTEXT`'
        self.query_span = '`QUERY`'
        self.answer_span = '`ANSWER`'
        self.examined_span_type = 'context'

    def prepare_hallucinated_df_info(self, exp_name: str):
        """
        Processes the data by performing several steps including formatting contexts,
        finding hallucination indices, and generating hallucination tokens.
        This method performs the following steps:
        1. Calls the `_prestage` method to prepare the data.
        2. Calls the `_filter_error_problematic_spans` method to filter out problematic spans.
        3. Iterates over each row in the dataframe `self.df`:
            a. Formats the context and counts the number of prompt tokens.
            b. Finds indices of hallucinations in the formatted context.
            c. Generates a mask for hallucination tokens.
        4. Updates the dataframe `self.df` with new columns:
            - 'formatted_context': The formatted context for each row.
            - 'n_prompt_tokens': The number of prompt tokens for each row.
            - 'hallu_indices': The indices of hallucinations for each row.
            - 'hallu_tokens': The hallucination tokens mask for each row.
        """

        self._prestage()
        self._filter_error_problematic_spans()

        formatted_contexts, n_prompt_tokens, hallu_indicies, hallu_tokens = [], [], [], []
        context_starts, context_ends = [], []
        query_starts, query_ends = [], []

        for i, row in self.df.iterrows():

            formatted_prompt, n_p_tokens, formatted_context = self._format_context(row)

            n_prompt_tokens.append(n_p_tokens)
            formatted_contexts.append(formatted_context)

            offset_mapping = self._llm_rg.tokenizer(
                formatted_context,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
                return_offsets_mapping=True)['offset_mapping'][0].tolist()

            hallu_idx = self._find_indices(
                response=formatted_context, 
                spans=row['problematic_spans']
            )

            hallu_indicies.append(hallu_idx)

            hallu_mask = self._find_hallu_tokens(offset_mapping, hallu_idx)
            hallu_tokens.append(hallu_mask)

            (context_start, context_end), (query_start, query_end) = self._find_context_query_token_idx(
                prompt_text=formatted_prompt, 
                offset_mapping=offset_mapping
            )

            context_starts.append(context_start)
            context_ends.append(context_end)
            query_starts.append(query_start)
            query_ends.append(query_end)

        for col_name, col_values in zip(
            ['formatted_context', 'n_prompt_tokens', 'hallu_indices', 'hallu_tokens'], 
            [formatted_contexts, n_prompt_tokens, hallu_indicies, hallu_tokens]):

            self.df[col_name] = col_values

        for col_name, col_values in zip(
            ['context_start_idx', 'context_end_idx', 'query_start_idx', 'query_end_idx'],
            [context_starts, context_ends, query_starts, query_ends]):

            self.df[col_name] = col_values

        # if problem comment out the following line
        # self.df.to_parquet(os.path.join(exp_name, 'hallu_df_info.parquet'))
        return self.df
    
    def _setup(self):

        if self._llm_rg.tokenizer is None:

            import transformers
            self._llm_rg.tokenizer = transformers.AutoTokenizer.from_pretrained(self._llm_rg.model_id)

    def _prestage(self):

        self.df.index = self.df[self._llm_rg.id_col]
        self.df.drop(columns=[self._llm_rg.id_col], inplace=True)
        self.df.index.name = None
        self.df = self.df.dropna()

        if 'question' in self.df.columns:
            self.df = self.df.rename(columns={'question': 'query'})

    def _filter_error_problematic_spans(
        self,
    ):
        """
        Filters out rows in the DataFrame where problematic spans cannot be found in the model response.

        This method adds an 'indices' column to the DataFrame, which contains the indices of the problematic spans
        within the model response for each row. It then removes rows where any of the indices are (None, None),
        indicating that the problematic span could not be found. Finally, it drops the 'indices' column from the DataFrame.

        Returns:
            None
        """

        self.df = self.df.dropna(how='all')
        self.df['indices'] = self.df.apply(lambda row: self._find_indices(row['model_response'], row['problematic_spans']), axis=1)
        self.df = self.df.loc[~self.df.index.isin([i for i, row in self.df.iterrows() if any(index == (None, None) for index in row['indices'])])]
        self.df = self.df.drop(columns=['indices'])

    @staticmethod
    def _find_indices(response: str, spans: List[str] | str) -> List[Tuple[int, int]]:
        """
        Function to find the starting and ending indices of each substring in 'problematic_spans' within 'model_response'.
        Returns a list of tuples (start, end) for each problematic span.
        If a span cannot be found, it returns (None, None).
        """

        indices = []
        if isinstance(spans, str):
            spans = [spans]

        if spans is None:
            return None
        
        for span in spans:
            start = response.rfind(span)
            if start != -1:
                end = start + len(span)
                indices.append((start, end))
            else:
                indices.append((None, None))
        
        return indices

    def _format_context(self, row):

        formatted_prompt = self._llm_rg._get_ready_prompt(
            row=row,
            prompt_columns=['query', 'context']
        )

        inputs = self._llm_rg.tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)
        n_prompt_tokens = inputs['input_ids'].shape[1]

        formatted_context = f"{self._llm_rg.tokenizer.decode(inputs['input_ids'][0])}{row['model_response']}"
        return formatted_prompt, n_prompt_tokens, formatted_context

    def _find_hallu_tokens(self, offset_mapping: str, hallu_indices: List[Tuple[int, int]]) -> List[bool]:
        
        hallu_tokens = [0] * len(offset_mapping)

        for idx, (start, end) in enumerate(offset_mapping):

            for hallu_start, hallu_end in hallu_indices:
                if start >= hallu_start and end <= hallu_end:
                    hallu_tokens[idx] = 1

        return hallu_tokens
    
    def set_config(
            self,
            context_span: str = '`CONTEXT`:',
            query_span: str = '`QUERY`:',
            answer_span: str = '`ANSWER`:',
            examined_span_type: str = 'context'
    ):
        
        self.context_span = context_span
        self.query_span = query_span
        self.answer_span = answer_span
        self.examined_span_type = examined_span_type
    
    def _find_context_query_char_idx(self, prompt_text: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:

        context_start = self._find_indices(prompt_text, self.context_span)[0][1]
        context_end = self._find_indices(prompt_text, self.query_span)[0][0]

        query_start = self._find_indices(prompt_text, self.query_span)[0][1]
        query_end = self._find_indices(prompt_text, self.answer_span)[0][0]

        return (context_start, context_end), (query_start, query_end)
    
    def _find_context_query_token_idx(self, prompt_text: str, offset_mapping) -> Tuple[Tuple[int, int], Tuple[int, int]]:

        context_token_start_idx, context_token_end_idx = None, None
        query_token_start_idx, query_token_end_idx = None, None

        (context_start, context_end), (query_start, query_end) = self._find_context_query_char_idx(prompt_text)

        for idx, (start, end) in enumerate(offset_mapping):

            if (start <= context_start <= end):
                context_token_start_idx = idx

            if (start <= context_end <= end) and (context_token_end_idx is None):
                context_token_end_idx = idx

            if (start <= query_start <= end):
                query_token_start_idx = idx
            
            if (start <= query_end <= end) and (query_token_end_idx is None):
                query_token_end_idx = idx

        return (context_token_start_idx, context_token_end_idx), (query_token_start_idx, query_token_end_idx)
    
    @staticmethod
    def agg_att(x: np.ndarray, f1_agg: str = 'mean', f2_agg: str = 'mean'):

        agg_funcs = {
            'mean': lambda x: np.mean(x, axis=-1),
            'max': lambda x: np.max(x, axis=-1),
            'min': lambda x: np.min(x, axis=-1),
            'median': lambda x: np.median(x, axis=-1),
        }

        return agg_funcs[f1_agg](agg_funcs[f2_agg](x))
    
    @staticmethod
    def prep_att_pipe(
        att_path: str,
        n_first_tokens: int = None,
        skip_first_n_tokens: int = None,
        skip_last_n_tokens: int = None,
        n_context_tokens_start_idx: int = None,
        n_context_tokens_end_idx: int = None,
        window_size: int = 0,
        window_step: int = 4,
        postprocess_fn: callable = None,
        valid_example_th: int = 4,
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Function to prepare the attention tensor for further analysis.
        It removes the prompt tokens and the last offset_size tokens from the context.
        """

        att_tensor = np.load(att_path)

        skip_first_n_tokens = skip_first_n_tokens if skip_first_n_tokens is not None else 0
        skip_last_n_tokens = skip_last_n_tokens if skip_last_n_tokens is not None else 0
        n_first_tokens = n_first_tokens if n_first_tokens is not None else att_tensor.shape[-2]

        att_tensor = att_tensor[..., slice(skip_first_n_tokens, n_first_tokens + skip_first_n_tokens - skip_last_n_tokens), slice(n_context_tokens_start_idx, n_context_tokens_end_idx)]
        
        if att_tensor.shape[-2] < valid_example_th:
            return None

        if (window_size) and (att_tensor.shape[-2] > window_size):

            att_tensor = {
                tuple([i, i + window_size]) : postprocess_fn(att_tensor[..., i: i + window_size, :], **kwargs) if kwargs else postprocess_fn(att_tensor[..., i: i + window_size, :])
                for i in range(0, att_tensor.shape[-2], window_step) if i + window_size <= att_tensor.shape[-2]
            }

        else:
            att_tensor = postprocess_fn(att_tensor, **kwargs) if kwargs else postprocess_fn(att_tensor)

        return att_tensor
    
    @staticmethod
    def load_attension_tensor(
        row: pd.Series,
        att_path: str,
        idx: str,
        examined_span_type: str = 'context',
        n_first_tokens: int = None,
        skip_first_n_tokens: int = None,
        skip_last_n_tokens: int = None,
        postprocess_fn: callable = None,
        window_size: int = 0,
        window_step: int = 4,
        valid_example_th: int = 4,
        ) -> Tuple[np.ndarray, str, str]:

        att_file = f"{idx}.npy"
        print(f"att_file: {att_file}")

        n_context_tokens_start_idx = row.get(f'{examined_span_type}_start_idx', None)
        n_context_tokens_end_idx = row.get(f'{examined_span_type}_end_idx', None)

        att_file_path = os.path.join(att_path, att_file)

        try:

            att_tensor = HallucinationDatasetExtractor.prep_att_pipe(
                att_path=att_file_path,
                n_first_tokens=n_first_tokens,
                skip_first_n_tokens=skip_first_n_tokens,
                skip_last_n_tokens=skip_last_n_tokens,
                n_context_tokens_start_idx=n_context_tokens_start_idx,
                n_context_tokens_end_idx=n_context_tokens_end_idx,
                postprocess_fn=postprocess_fn,
                window_size=window_size,
                window_step=window_step,
                valid_example_th=valid_example_th
            )
            
            return att_tensor, None, att_file
        
        except Exception as e:
            print(f"Error: {e}")
            return None, att_file, att_file
        
    def prepare_and_load_att_tensors(
            self,
            examined_span_type: str = 'context',
            n_first_tokens: int = 8,
            skip_first_n_tokens: int = 8,
            skip_last_n_tokens: int = 0,
            window_size: int = 0,
            agg_func: callable = None,
            window_step: int = 4,
            valid_example_th: int = 4
        ):
        
        X, errors, not_valid = {}, [], []

        with ProcessPoolExecutor() as executor:

            futures = [
                executor.submit(
                    self.load_attension_tensor,
                    row=row,
                    att_path=self.att_dir_path,
                    idx=idx,
                    skip_first_n_tokens=skip_first_n_tokens,
                    skip_last_n_tokens=skip_last_n_tokens,
                    postprocess_fn=agg_func,
                    window_size=window_size,
                    examined_span_type=examined_span_type,
                    n_first_tokens=n_first_tokens,
                    window_step=window_step,
                    valid_example_th=valid_example_th
                )
                for idx, row in self.df.iterrows()
            ]
            
            for future in as_completed(futures):

                result, error, att_file = future.result()
                
                if error:
                    errors.append(error)
                else:

                    if result is not None:
                        X[att_file] = result
                    else:
                        not_valid.append(att_file)

        print(f"Completed with {len(errors)} errors and {len(not_valid)} invalid examples.")
        return X, errors

    def prepare_hallu_labels(
            self,
            X: dict,
            n_first_tokens: int = 8,
            att_file_type: str = 'npy',
    ):

        Y = {}

        for i, row in self.df.iterrows():

            att_file = f"{i}.{att_file_type}"

            if att_file not in X:
                continue

            n_prompt_tokens = row['n_prompt_tokens']
            hallu_tokens = np.array(row['hallu_tokens'])

            if isinstance(X.get(att_file), dict):

                y = {
                    k: hallu_tokens[n_prompt_tokens + k[0]: n_prompt_tokens + k[1]].any().astype(int)
                    for k in X[att_file].keys()
                }

            else:

                idx = slice(n_prompt_tokens, n_first_tokens) \
                    if n_first_tokens is None \
                    else slice(n_prompt_tokens, n_prompt_tokens + n_first_tokens)
                
                y = hallu_tokens[idx].any().astype(int)

            Y[att_file] = y

        return Y
    
    def _flatten_windowed_dataset(
        self,
        X: dict,
        Y: dict
    ):

        X_flat, Y_flat = {}, {}

        for att_file, att_tensor in X.items():

            if isinstance(att_tensor, dict):

                for k, v in att_tensor.items():
                    X_flat[f"{att_file}__{k}"] = v
                    Y_flat[f"{att_file}__{k}"] = Y[att_file][k]

            else:
                X_flat[att_file] = att_tensor
                Y_flat[att_file] = Y[att_file]

        del X, Y
        return X_flat, Y_flat
    
    @staticmethod
    def _prep_att_dataset(
            X, 
            Y, 
            att_file_type: str = 'npy',
            label_column: str = 'label'):

        sorted_x_keys = sorted(X.keys())
        dataset_names = [
            s.split('__')[0] 
            .removesuffix(f'.{att_file_type}') 
            .rsplit('_', 1)[0] for s in sorted_x_keys
        ]

        X = np.array([X[k] for k in sorted_x_keys])
        Y = np.array([Y[k] for k in sorted_x_keys])

        df = pd.DataFrame(X.reshape(X.shape[0], -1))
        df[label_column] = Y.tolist()
        df['dataset'] = dataset_names

        return df

    def create_attension_dataset(
            self,
            exp_name: str,
            examined_span_type: str = 'context',
            n_first_tokens: int = 8,
            skip_first_n_tokens: int = 8,
            skip_last_n_tokens: int = 0,
            att_file_type: str = 'npy',
            save_to_disk: bool = True,
            saving_name_params: dict = {
                'path': None,
                'att_ds_name': 'att_ds',
                'start': 0,
                'end': None
            },
            window_size: int = 0,
            window_step: int = 4,
            valid_example_th: int = 4,
            agg_func: callable = None
            ):
        
        if agg_func is None:
            agg_func = self.agg_att
        

        X, _ = self.prepare_and_load_att_tensors(
            examined_span_type=examined_span_type,
            n_first_tokens=n_first_tokens,
            skip_first_n_tokens=skip_first_n_tokens,
            skip_last_n_tokens=skip_last_n_tokens,
            window_size=window_size,
            agg_func=agg_func,
            window_step=window_step,
            valid_example_th=valid_example_th
        )

        Y = self.prepare_hallu_labels(X, n_first_tokens=n_first_tokens, att_file_type=att_file_type)

        if window_size:
            X, Y = self._flatten_windowed_dataset(X, Y)

        df = self._prep_att_dataset(X, Y, att_file_type=att_file_type)

        if save_to_disk:

            dest_path = os.path.join(
                saving_name_params.get('path', '.'),
                exp_name, 
                saving_name_params.get('att_ds_name', 'att_ds')
            )

            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            metadata = {
                'examined_span_type': examined_span_type,
                'n_first_tokens': n_first_tokens,
                'skip_first_n_tokens': skip_first_n_tokens,
                'skip_last_n_tokens': skip_last_n_tokens,
                'window_size': window_size,
                'window_step': window_step,
                'valid_example_th': valid_example_th,
                'agg_func': agg_func.__name__,
                'att_file_type': att_file_type,
            }

            filename = f'attension_{saving_name_params.get("start", 0)}_{saving_name_params.get("end", len(df))}.parquet'
            df.to_parquet(os.path.join(dest_path, filename))

            with open(os.path.join(dest_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
        return df

    def save_data(self):
        self.df.to_parquet(self.output_path)

    def run(self):
        self.process_data()
        self.save_data()
