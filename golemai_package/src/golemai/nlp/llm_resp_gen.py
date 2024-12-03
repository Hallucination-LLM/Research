import ast
import json
import logging
import os
import pickle
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from golemai.io.file_ops import (
    load_json,
    read_file_to_df,
    save_json,
    save_numpy,
    save_pickle,
)
from golemai.nlp.gpu_ops import setup_environment
from golemai.nlp.llm_module import (
    generate_response_api,
    generate_response_api_prase,
    generate_response_local,
    load_generation_config,
    load_model,
    prepare_prompt,
    run_model_sanity_check,
)
from openai import OpenAI
from tqdm import tqdm

if torch.cuda.is_available():

    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

from golemai.nlp.prompts import (
    QUERY_INTRO_FEWSHOT,
    QUERY_INTRO_NO_ANS,
    SYSTEM_MSG_RAG,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMRespGen:
    """
    LLMRespGen is a class for generating responses using various types of language models (local, API-based).
    
    It can iterate over a DataFrame and generate responses for each row based on the specified columns.
    It enables to configure the model, generation settings, and checkpointing for the responses.

    Attributes:
        df (pd.DataFrame, optional): Input DataFrame containing data for processing.
        model_type (Literal["local", "api", "openai"]): Type of model to use.
        api_url (str, optional): URL for API-based models.
        api_key (str, optional): API key for authentication.
        prompt_template (str, optional): Template for formatting prompts.
        system_msg (str, optional): System message for the model.
        checkpoint_path (str, optional): Path to model checkpoint.
        device_num (int): GPU device number for local models.
        batch_size (int): Batch size for processing.
        has_system_role (bool): Whether the model supports system role.
        timeout (int): Timeout duration in seconds.
        device (str): Device to use for local models.
        client (OpenAI, optional): Client for API-based models.
        model_config (dict, optional): Configuration for the model.
        model_id (str): Identifier for the model.
        llm (Model, optional): Loaded language model.
        tokenizer (Tokenizer, optional): Tokenizer for the language model.
        generation_config (dict, optional): Configuration for generating responses.
        _prefix_funcs (dict): Dictionary of prefix functions.
        model_responses (list, optional): List of model responses.
        att_hidden_config (dict, optional): Configuration for attention and hidden states gathering.
    """
    
    def __init__(
        self,
        id_col: str,
        df: Optional[pd.DataFrame] = None,
        model_type: Literal["local", "api", "openai"] = "api",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        system_msg: Optional[str] = "",
        checkpoint_path: Optional[str] = None,
        device_num: int = 0,
        batch_size: int = 1,
        has_system_role: bool = False,
        timeout: int = 30,
    ) -> None:

        """
        Initialize the LLMRespGen (LLM Response Generator) class.

        Args:
            df (pd.DataFrame, optional): Input DataFrame containing data for processing. Defaults to None.
            model_type (Literal["local", "api", "openai"]): Type of model to use. Defaults to "api".
            api_url (str, optional): URL for API-based models. Defaults to None.
            api_key (str, optional): API key for authentication. Defaults to None.
            prompt_template (str, optional): Template for formatting prompts. Defaults to None.
            system_msg (str, optional): System message for the model. Defaults to empty string.
            checkpoint_path (str, optional): Path to model checkpoint. Defaults to None.
            device_num (int): GPU device number for local models. Defaults to 0.
            batch_size (int): Batch size for processing. Defaults to 1.
            has_system_role (bool): Whether the model supports system role. Defaults to False.
            timeout (int): Timeout duration in seconds. Defaults to 30.
        """
        super(LLMRespGen, self).__init__()

        self.df = df.reset_index(drop=True) if df is not None else None
        self._set_model_type(model_type)

        self.device = (
            setup_environment(device_num)
            if self.model_type == "local"
            else "cpu"
        )

        self.device_num = device_num

        self.api_url = api_url
        self.api_key = api_key
        self.client = self._init_client()

        self.timeout = timeout

        self.prompt_template = prompt_template
        self.system_msg = system_msg
        self.batch_size = batch_size
        self.has_system_role = has_system_role
        self.skip_prompt_tokens = True
        self.skip_special_tokens = True

        self.checkpoint_path = None
        self.configure_checkpoint(checkpoint_path=checkpoint_path)

        self.formatted_column = None
        self.prompt_columns = None

        self.model_config = None
        self.model_id = "unsloth/gemma-2-9b-it-bnb-4bit"
        self.llm = None
        self.tokenizer = None
        self.generation_config = None
        self._prefix_funcs = {}

        self.model_responses = None
        self.att_hidden_config = {}
        self.id_col = id_col

    def _set_model_type(
        self, model_type: Literal["local", "api", "openai"]
    ) -> None:
        """
        Set the model type to either 'local' or 'api' or 'openai'.

        Args:
            model_type (Literal["local", "api", "openai"]): The type of model to use.

        Raises:
            ValueError: If an invalid model type is provided.
        """

        logger.debug(f"_set_model_type: {model_type}")

        if model_type not in ["local", "api", "openai"]:
            logger.error(
                "Invalid model type. Choose either 'local' or 'api' or 'openai'."
            )
            raise ValueError(
                "Invalid model type. Choose either 'local' or 'api' or 'openai'."
            )

        self.model_type = model_type

    def _init_client(self) -> None:
        """
        Initialize the client for API model. Destination host should
        be compatible with OpenAI API. For example model could be
        hosted by using vLLM server.

        Raises:
            ValueError: If API URL or API key is not provided.
        """

        client = None

        if self.model_type in ["api", "openai"]:

            if self.api_url is None:
                error_msg = "API URL must be provided for API model."
                logger.error(error_msg)
                raise ValueError(error_msg)

            if self.api_key is None:
                error_msg = "API key must be provided for API model."
                logger.error(error_msg)
                raise ValueError(error_msg)

            client = OpenAI(base_url=self.api_url, api_key=self.api_key)

        return client

    def load_llm(
        self, model_id: str = "unsloth/gemma-2-9b-it-bnb-4bit", **kwargs
    ) -> None:
        """
        Load the Language Model (LLM) with the specified model ID and configuration.
        Args:
            model_id (str, optional): The ID of the model to load. Defaults to 'unsloth/gemma-2-9b-it-bnb-4bit'.
            **kwargs: Additional keyword arguments for configuring the LLM.
        Returns:
            None
        """

        logger.debug(f"_load_llm: {model_id = }, {kwargs = }")

        DEFAULT_LLM_CONFIG = {
            "model_id": model_id,
            "max_seq_length": 8192,
            "load_in_4bit": True,
            "device_map": self.device_num
            if self.device_num == "auto"
            else {"": f"{self.device}"},
            "dtype": torch.bfloat16,
            "use_unsloth": True,
            # "attn_implementation": "flash_attention_2"
        }

        if kwargs is None:
            kwargs = {}

        if self.model_type in ["api", "openai"]:
            logger.warning(
                "Model type is 'api'. To load model, first set model type to 'local'."
            )
            return None

        if (self.llm is not None) and (
            self.model_config == (kwargs | {"model_id": model_id})
        ):
            logger.warning(
                f"Model already loaded with config: {self.model_config}"
            )
            return None

        if kwargs is None:
            self.model_config = DEFAULT_LLM_CONFIG
        else:
            self.model_config = DEFAULT_LLM_CONFIG | kwargs

        self.model_id = model_id
        self.llm, self.tokenizer = load_model(**self.model_config)

        run_model_sanity_check(
            model=self.llm, tokenizer=self.tokenizer, device=self.device
        )

        self.set_generation_config()

        return self

    def set_generation_config(self, model_id: str = None, **kwargs) -> None:
        """
        Set the generation configuration for the model.
        This method configures the generation settings for the model based on the
        model type (local or API). It supports both local and API-based models
        (e.g., OpenAI). The configuration can be customized using keyword arguments.
        Args:
            model_id (str, optional): The identifier for the model. Defaults to None.
            **kwargs: Additional keyword arguments to customize the generation
                      configuration.
        Raises:
            ValueError: If the model is not loaded when setting the configuration
                        for a local model.
        Returns:
            None
        """

        logger.debug(f"set_generation_config: {kwargs = }")

        DEFAULT_LOCAL_GENERATION_CONFIG = {
            "max_new_tokens": 1024,
            # "temperature": 0.0,
            "repetition_penalty": 1.0,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
        }

        DEFAULT_API_GENERATION_CONFIG = {
            "max_tokens": 1024,
            "temperature": 0.0,
            "timeout": 20,
        }

        if "skip_prompt_tokens" in kwargs:
            self.skip_prompt_tokens = kwargs.pop("skip_prompt_tokens")

        if "skip_special_tokens" in kwargs:
            self.skip_special_tokens = kwargs.pop("skip_special_tokens")

        if self.model_type in ["api", "openai"]:

            logger.warning(
                "Model type is 'api'. Setting generation config for API."
            )

            if model_id is not None:
                self.model_id = model_id

            self.generation_config = (
                DEFAULT_API_GENERATION_CONFIG
                if kwargs is None
                else DEFAULT_API_GENERATION_CONFIG | kwargs
            )

            return self

        if self.llm is None:
            logger.debug(f"Model instance: {type(self.llm)}")
            logger.error("Model not loaded. Load model first.")
            raise ValueError("Model not loaded. Load model first.")

        self.generation_config = load_generation_config(
            model_id=self.model_id,
            **(
                DEFAULT_LOCAL_GENERATION_CONFIG
                if kwargs is None
                else DEFAULT_LOCAL_GENERATION_CONFIG | kwargs
            ),
        )

        logger.debug(
            f"Generation config set: {self.generation_config.to_dict()}"
        )

        return self

    def set_prompt_template(self, prompt_template: str) -> None:
        """
        Sets the prompt template for the language model response generator.
        Args:
            prompt_template (str): The template string to be used for generating prompts.
        Returns:
            None
        """

        logger.debug(f"set_prompt_template: {prompt_template = }")
        self.prompt_template = prompt_template
        return None

    def set_system_msg(self, system_msg: str) -> None:
        """
        Sets the system message for the instance.
        Args:
            system_msg (str): The system message to be set.
        Returns:
            None
        """

        logger.debug(f"set_system_msg: {system_msg = }")
        self.system_msg = system_msg
        return None

    def get_prompt_template(self) -> str:

        logger.debug(f"get_prompt_template: {self.prompt_template = }")
        return self.prompt_template

    def _prepare_messages(
        self,
        user_input: str,
        messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Prepares a list of messages for the language model response generation.
        Args:
            user_input (str): The input message from the user.
            messages (List[Dict[str, str]], optional): A list of existing messages. Defaults to None.
        Returns:
            List[Dict[str, str]]: The updated list of messages including the user input and optionally a system message.
        """

        if messages is None:
            messages = []

        if self.has_system_role:
            messages.append({"role": "system", "content": self.system_msg})
        else:
            user_input = f"{self.system_msg} {user_input}"

        messages.append({"role": "user", "content": user_input})

        logger.debug(f"_prepare_messages: {messages = }")

        return messages

    def prepare_prompt(self, **kwargs) -> str:
        """
        Prepares a prompt string by formatting the prompt template with the given keyword arguments.
        Args:
            **kwargs: Arbitrary keyword arguments to be used for formatting the prompt template.
        Returns:
            str: The formatted prompt string.
        """

        return self.prompt_template.format(**kwargs)

    def _get_ready_prompt(
        self,
        row: pd.Series,
        formatted_column: Optional[str] = None,
        prompt_columns: Optional[Union[str, list[str]]] = None,
    ) -> str:
        """
        Generates a ready-to-use prompt based on the provided row data and specified columns.
        Args:
            row (pd.Series): A pandas Series object containing the data for the current row.
            formatted_column (Optional[str]): The name of the column containing pre-formatted prompt text. If provided, this column's value will be used as the prompt.
            prompt_columns (Optional[Union[str, list[str]]]): The name(s) of the columns to be used for generating the prompt. If `formatted_column` is not provided, these columns will be used to prepare the prompt.
        Returns:
            str: The generated prompt, ready for use with the language model.
        """

        if formatted_column is not None:
            prompt = row[formatted_column]

        else:

            prompt = self.prepare_prompt(
                **{col: row[col] for col in prompt_columns}
            )

        return (
            prepare_prompt(
                tokenizer=self.tokenizer,
                user_input=prompt,
                system_input=self.system_msg,
                has_system_role=self.has_system_role,
            )
            if self.model_type == "local"
            else prompt
        )

    def configure_checkpoint(
        self,
        checkpoint_path: str = None,
        checkpoint_time_format: str = "%Y-%m-%d_%H-%M-%S",
        checkpoint_freq: int = 5,
    ) -> None:
        """
        Configures the checkpoint settings for the model.
        Args:
            checkpoint_path (str, optional): The path where checkpoints will be saved. Defaults to None.
            checkpoint_time_format (str, optional): The format for the timestamp used in checkpoint filenames. Defaults to "%Y-%m-%d_%H-%M-%S".
            checkpoint_freq (int, optional): The frequency (in epochs) at which checkpoints will be saved. Defaults to 25.
        Returns:
            None
        """

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path

        if checkpoint_time_format is not None:
            self.checkpoint_time_format = checkpoint_time_format

        if checkpoint_freq is not None:
            self.checkpoint_freq = checkpoint_freq

        return self

    def _load_checkpoint(
        self,
        row_start: int = 0,
        row_end: int = None,
        model_response_col: str = "model_responses",
    ) -> dict:
        """
        Loads a checkpoint from a specified path or initializes a new checkpoint if the path does not exist.
        Args:
            row_start (int, optional): The starting row index for processing. Defaults to 0.
            row_end (int, optional): The ending row index for processing. Defaults to None.
            model_response_col (str, optional): The column name for storing model responses. Defaults to "model_responses".
        Returns:
            dict: A dictionary containing the model responses and configuration details.
        """

        if (self.checkpoint_path is None) or (
            not os.path.exists(self.checkpoint_path)
        ):

            model_responses = {
                "model_id": self.model_id,
                "device": f"{self.device}"
                if self.device == "auto"
                else f"{self.device}:{self.device_num}",
                "row_start": row_start,
                "row_end": row_end,
                "eval_time": None,
                "timestamp": None,
                "system_message": self.system_msg,
                "prompt_template": self.prompt_template,
                "model_config": self.model_config,
                "generation_config": self.generation_config.to_diff_dict(),
                model_response_col: {},
            }

            if "dtype" in self.model_config:
                model_responses["model_config"]["dtype"] = str(
                    self.model_config["dtype"]
                )

        else:

            logger.debug(f"Loading checkpoint from {self.checkpoint_path}")
            model_responses = load_json(self.checkpoint_path)
            row_start = max(
                self.df.loc[self.df[self.id_col] == list(model_responses.get(model_response_col).keys())[-1]].index[0] + 1,
                row_start,
            )
            logger.debug(f"Starting from row {row_start}")

        self.model_responses = model_responses
        return row_start, row_end

    def _save_checkpoint(
        self,
        eval_run_name: str,
        checkpoint_run_name: str,
        eval_start_time: float,
        last_checkpoint: bool = False,
    ) -> None:
        """
        Saves a checkpoint of the model responses to a JSON file.
        Args:
            eval_run_name (str): The name of the evaluation run.
            eval_start_time (float): The start time of the evaluation run.
            last_checkpoint (bool, optional): Flag indicating if this is the last checkpoint. Defaults to False.
        Returns:
            None
        """

        logger.debug(
            f"_save_checkpoint: {eval_run_name = }, {last_checkpoint = }"
        )

        timestamp = datetime.now().strftime(self.checkpoint_time_format)
        eval_end_time = perf_counter()

        checkpoint_path = (
            self.checkpoint_path
            if self.checkpoint_path is not None
            else os.path.join(
                ".",
                eval_run_name,
                "checkpoints",
                f"{checkpoint_run_name}_{'checkpoint' if not last_checkpoint else timestamp}.json",
            )
        )

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))

        self.checkpoint_path = checkpoint_path
        self.model_responses["eval_time"] = eval_end_time - eval_start_time
        self.model_responses["timestamp"] = timestamp

        save_json(self.model_responses, self.checkpoint_path)

    def _prepare_json_func(self, json_schema: dict) -> callable:
        """
        Prepares a callable function based on the provided JSON schema.
        This method checks if a prefix function for the given JSON schema is already cached.
        If it is, the cached function is returned. Otherwise, a new prefix function is created,
        cached, and then returned.
        Args:
            json_schema (dict): The JSON schema to be used for creating the prefix function.
        Returns:
            callable: A function that can be used as a prefix function for the given JSON schema.
        """

        json_schema_str = json.dumps(json_schema, ensure_ascii=False)

        if json_schema_str in self._prefix_funcs:
            logger.debug(
                f"Using cached prefix function for schema: {json_schema} - number of saved functions: {len(self._prefix_funcs)}"
            )
            prefix_function = self._prefix_funcs[json_schema_str]

        else:
            parser = JsonSchemaParser(json_schema)
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                self.tokenizer, parser
            )
            self._prefix_funcs[json_schema_str] = prefix_function

        return prefix_function

    def _generate_llm_response(
        self,
        inputs: Union[str, list[str]],
        json_schema: Optional[dict] = None,
        pydantic_model: Optional[object] = None,
    ) -> str:
        """
        Generates a response from a language model (LLM) based on the provided inputs.
        Based on the model type (local or API), the response is generated using the appropriate method.
        Args:
            inputs (Union[str, list[str]]): The input text or list of input texts to generate a response for.
            json_schema (Optional[dict], optional): A JSON schema to guide the response generation. Defaults to None.
            pydantic_model (Optional[object], optional): A Pydantic model to parse the response into. Defaults to None.
        Returns:
            str: The generated response from the LLM.
        Raises:
            ValueError: If the model and tokenizer are not set for a local LLM.
            ValueError: If the host is not provided for an API LLM.
            ValueError: If batch_size > 1 is used for an API LLM.
        """

        logger.debug(f"generate_llm_response: {len(inputs) = }")

        if self.model_type == "local":

            if (self.llm is None) or (self.tokenizer is None):
                error_msg = "Model and tokenizer must be set for local LLM"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if json_schema is not None:
                prefix_function = self._prepare_json_func(json_schema)
                self.generation_config.prefix_allowed_tokens_fn = (
                    prefix_function
                )

            output = generate_response_local(
                inputs=inputs,
                model=self.llm,
                tokenizer=self.tokenizer,
                generation_config=self.generation_config,
                skip_prompt_tokens=self.skip_prompt_tokens,
                device=self.device,
            )

            return output

        else:

            if self.api_url is None:
                logger.error(f"Host must be provided for API LLM")
                raise ValueError("Host must be provided for API LLM")

            if self.batch_size > 1:
                error_msg = (
                    "Currently, batch_size > 1 is not supported for API LLM"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            extra_body = None

            if json_schema is not None:

                extra_body = {
                    "guided_json": json_schema,
                    "guided_decoding_backend": "lm-format-enforcer",
                }

            if self.model_type == "openai":
                if json_schema is not None:
                    json_schema["additionalProperties"] = False

                    extra_body = {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "json_output",
                                "strict": True,
                                "schema": json_schema,
                            },
                        }
                    }

            if pydantic_model is not None:
                return generate_response_api_prase(
                    client=self.client,
                    model_id=self.model_id,
                    messages=self._prepare_messages(
                        user_input=inputs[0]
                        if isinstance(inputs, list)
                        else inputs
                    ),
                    response_format=pydantic_model,
                    **self.generation_config,
                )

            return generate_response_api(
                client=self.client,
                model_id=self.model_id,
                messages=self._prepare_messages(
                    user_input=inputs[0] if isinstance(inputs, list) else inputs
                ),
                extra_body=extra_body if json_schema is not None else None,
                **self.generation_config,
            )

    def _get_run_name(
        self,
        eval_run_name: Optional[str] = None,
        row_start: int = 0,
        row_end: int = None,
    ) -> str:

        eval_run_name = (
            eval_run_name if eval_run_name is not None else "llm_evaluation"
        )
        eval_run_name = f"{eval_run_name}_{row_start}_{row_end if row_end is not None else self.df.shape[0]}"

        return eval_run_name

    def _process_batch(
        self, batch_input: list, json_schemas: Optional[list] = None
    ):

        logger.debug(f"_process_batch: {len(batch_input) = }")

        if json_schemas and len(set(json_schemas)) > 1:

            warning_msg = "All JSON schemas in a batch must be the same - processing one at a time."
            logger.warning(warning_msg)
            llm_responses = []

            for input_, json_schema in zip(batch_input, json_schemas):

                output = self._generate_llm_response(
                    inputs=input_, json_schema=ast.literal_eval(json_schema)
                )

                if isinstance(output, list):
                    llm_responses.extend(output)
                else:
                    llm_responses.append(output)

                torch.cuda.empty_cache()
                del output

        else:

            json_schema = (
                ast.literal_eval(json_schemas[0]) if json_schemas else None
            )
            llm_responses = self._generate_llm_response(
                inputs=batch_input, json_schema=json_schema
            )
            torch.cuda.empty_cache()

        return (
            [llm_responses]
            if isinstance(llm_responses, (str))
            else llm_responses
        )
    
    def configure_att_hidden_config(
        self,
        take_only_generated: bool = True,
        prompt_offset: int = 0,
    ) -> None:
        """
        Sets the configuration for processing attention weights and hidden states.
        Args:
            take_only_generated (bool): Whether to take only the generated tokens. Defaults to True.
            prompt_offset (int): The offset for taking tokens from the prompt. Defaults to 0.
        Returns:
            None
        """

        logger.debug(f"configure_att_hidden_config: {take_only_generated = }, {prompt_offset = }")
        self.att_hidden_config = {
            "take_only_generated": take_only_generated,
            "prompt_offset": prompt_offset,
        }

        return None

    def _process_hidden_or_att(
        self,
        x,
    ) -> np.ndarray:
        """
        Processes hidden states or attention weights from a tuple of tensors.
        Args:
            x (Tuple[Tuple[torch.Tensor]]): A tuple of tuples containing torch tensors.
                Each inner tuple represents the hidden states or attention weights for a token.
        Returns:
            np.ndarray: A numpy array containing the processed hidden states or attention weights.
                The array is concatenated along the last dimension, with an additional dimension
                added for all but the first element in the input tuple.
        """

        n_generated_tokens = len(x) - 1

        n_layers = self.llm.config.num_hidden_layers
        n_heads = self.llm.config.num_attention_heads

        n_prompt_tokens = x[0][0].shape[-2]

        print(f"n_prompt_tokens: {n_prompt_tokens}, n_generated_tokens: {n_generated_tokens}")

        num_all_tokens = n_prompt_tokens + n_generated_tokens

        prompt_offset = self.att_hidden_config.get("prompt_offset", 0)
        take_only_generated = self.att_hidden_config.get("take_only_generated", True)

        init_matrix = np.zeros(shape=(
            n_layers, 
            n_heads,
            num_all_tokens, 
            num_all_tokens), 
        dtype=np.float16)

        for i, t in enumerate(x[0]):
            init_matrix[i, :, :n_prompt_tokens, :] = t.detach().cpu().to(torch.float16).numpy().squeeze()[..., :num_all_tokens]

        for i, token_att in enumerate(x[1:], start=0):
            for j, t in enumerate(token_att):
                init_matrix[j, :, n_prompt_tokens + i, :] = t.detach().cpu().to(torch.float16).numpy().squeeze()[..., :num_all_tokens]

        return init_matrix[:, :, (max(n_prompt_tokens - prompt_offset, 0)) if take_only_generated else 0:, :]


        # return np.concatenate(
        #     (
        #         first_init_matrix,
        #         np.stack(
        #             tuple(
        #                 [
        #                     t.detach().cpu().to(torch.float16).numpy().squeeze()
        #                     for t in token_att
        #                 ]
        #             ),
        #             axis=0,
        #         )[..., :num_all_tokens][..., np.newaxis, :]
        #         for i, token_att in enumerate(x[1:])
        #     ),
        #     axis=-2,
        # )


    def _process_df(
        self,
        formatted_column: Optional[str] = None,
        prompt_columns: Optional[Union[str, list[str]]] = None,
        json_schema_col: Optional[str] = None,
        row_start: int = 0,
        row_end: int = None,
        eval_run_name: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
        max_prompt_length_col: Optional[str] = None
    ) -> None:
        """
        Processes a DataFrame to generate model responses in batches.
        This method processes the DataFrame in batches, generating model responses for each batch.
        It handles the following steps:
        - Iterates over the specified rows of the DataFrame.
        - Prepares prompts based on the specified columns.
        - Checks if the prompt length exceeds the maximum allowed length and skips such rows.
        - Collects prompts and JSON schemas (if provided) into batches.
        - Generates model responses for each batch.
        - Processes and saves attentions and hidden states if available.
        - Saves checkpoints at specified intervals.
        - Returns the model responses.
        Args:
            formatted_column (Optional[str]): The name of the column containing formatted text.
            prompt_columns (Optional[Union[str, list[str]]]): The name(s) of the column(s) to be used as prompts.
            json_schema_col (Optional[str]): The name of the column containing JSON schemas.
            row_start (int): The starting index of the rows to process.
            row_end (int): The ending index of the rows to process.
            eval_run_name (Optional[str]): The name of the evaluation run.
            max_prompt_length (Optional[int]): The maximum allowed length for prompts.
            max_prompt_length_col (Optional[str]): The name of the column containing prompt lengths.
            row_id_col (Optional[str]): The name of the column containing row IDs.
        Returns:
            None
        """

        batch_input = []
        batch_ids = []
        json_schemas = []

        eval_start_time = perf_counter()

        if not os.path.exists(eval_run_name):
            os.makedirs(eval_run_name)

        checkpoint_eval_run_name = self._get_run_name(
            eval_run_name=eval_run_name, row_start=row_start, row_end=row_end
        )

        for i, row in tqdm(
            self.df.iloc[slice(row_start, row_end)].iterrows(),
            total=(row_end if row_end is not None else len(self.df))
            - row_start,
        ):
            
            id_ = row[self.id_col]

            prompt = self._get_ready_prompt(
                row=row,
                formatted_column=formatted_column,
                prompt_columns=prompt_columns,
            )

            if (max_prompt_length_col is not None) and (
                max_prompt_length is not None
            ):
                if row[max_prompt_length_col] > max_prompt_length:

                    logger.warning(
                        f"Skipping row {i} due to prompt length > {max_prompt_length}"
                    )
                    self.model_responses["model_responses"][id_] = "<SKIPPED>"
                    continue

            batch_input.append(prompt)
            batch_ids.append(id_)

            if json_schema_col is not None:
                json_schemas.append(row[json_schema_col])

            end_cond = (
                (i == row_end - 1)
                if row_end is not None
                else (i == len(self.df) - 1)
            )

            if (len(batch_input) == self.batch_size) or (end_cond):

                logger.debug(
                    f"Generating responses for batch with indicies: {i - self.batch_size + 1} - {i}"
                )

                output = self._process_batch(
                    batch_input=batch_input, json_schemas=json_schemas
                )

                if output is None:
                    llm_responses = ["<CUDA_ERROR>"] * len(batch_input)

                else:

                    llm_responses = (
                        output.sequences
                        if hasattr(output, "sequences")
                        else output
                    )

                    if isinstance(llm_responses, torch.Tensor):

                        llm_responses = self.tokenizer.batch_decode(
                            llm_responses, skip_special_tokens=self.skip_special_tokens
                        )

                    if hasattr(output, "attentions"):

                        if output.attentions is not None:

                            try:

                                start = perf_counter()
                                output.attentions = self._process_hidden_or_att(
                                    output.attentions
                                )
                                end = perf_counter()
                                print(f"Time taken to process attentions: {end - start:.3f} seconds")

                            except Exception as e:
                                logger.error(
                                    f"Error processing attentions: {e}"
                                )
                            else:
                                att_dir = os.path.join(
                                    eval_run_name, "attentions"
                                )

                                if not os.path.exists(att_dir):
                                    os.makedirs(att_dir)

                                try:
                                    
                                    save_numpy(
                                        output.attentions,
                                        file_path=os.path.join(
                                            att_dir, f"{id_}.npy"
                                        ),
                                    )

                                except Exception as e:
                                    logger.error(
                                        f"Error saving attentions: {e}"
                                    )

                    if hasattr(output, "hidden_states"):

                        if output.hidden_states is not None:

                            try:
                                output.hidden_states = (
                                    self._process_hidden_or_att(
                                        output.hidden_states
                                    )
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error processing hidden states: {e}"
                                )
                            else:

                                hs_dir = os.path.join(
                                    eval_run_name, "hidden_states"
                                )

                                if not os.path.exists(hs_dir):
                                    os.makedirs(hs_dir)

                            try:
                                
                                save_numpy(
                                    output.hidden_states,
                                    file_path=os.path.join(hs_dir, f"{id_}.npy"),
                                )

                            except Exception as e:
                                logger.error(f"Error saving hidden states: {e}")

                for idx, response in zip(batch_ids, llm_responses):
                    self.model_responses["model_responses"][idx] = response

                del batch_input
                torch.cuda.empty_cache()

                batch_input = []
                batch_ids = []

            if ((i % self.checkpoint_freq == 0) and (i > 0)) or (end_cond):

                self._save_checkpoint(
                    eval_run_name=eval_run_name,
                    checkpoint_run_name=checkpoint_eval_run_name,
                    eval_start_time=eval_start_time,
                    last_checkpoint=end_cond,
                )

        return self.model_responses

    def get_responses(
        self,
        eval_run_name: Optional[str] = None,
        formatted_column: Optional[str] = None,
        prompt_columns: Optional[Union[str, list[str]]] = None,
        json_schema_col: Optional[str] = None,
        row_start: int = 0,
        row_end: int = None,
        max_prompt_length_col: str = None,
        max_prompt_length: int = None,
    ) -> dict:
        """
        Generate responses based on the provided parameters. This method processes the input DataFrame
        and based on columns containing prompts and JSON schemas, generates responses using the language model.
        If formatted_column is provided, it will be used as the prompt. Otherwise, the prompt will be generated
        based on the specified prompt_columns.
        Args:
            eval_run_name (Optional[str]): Name of the evaluation run. Defaults to None.
            formatted_column (Optional[str]): Column name containing formatted prompts. Defaults to None.
            prompt_columns (Optional[Union[str, list[str]]]): Column(s) containing prompt data. Defaults to None.
            json_schema_col (Optional[str]): Column name containing JSON schema. Defaults to None.
            row_start (int): Starting row index for processing. Defaults to 0.
            row_end (int): Ending row index for processing. Defaults to None.
            max_prompt_length_col (str): Column name for maximum prompt length. Defaults to None.
            max_prompt_length (int): Maximum length of the prompt. Defaults to None.
            row_id_col (str): Column name for the index. Defaults to None.
        Returns:
            dict: A dictionary containing the generated responses.
        Raises:
            ValueError: If neither `prompt_template` nor `formatted_column` is provided.
            ValueError: If both `formatted_column` and `prompt_columns` are None.
        """

        logger.debug(
            f"get_responses: {formatted_column = }, {prompt_columns = }, {row_start = }, {row_end = }"
        )

        if (self.prompt_template is None) and (formatted_column is None):
            error_msg = (
                "Either prompt_template or formatted_column must be provided."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if all([formatted_column is None, prompt_columns is None]):
            error_msg = (
                "Either formatted_column or prompt_columns must be provided."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        row_start, row_end = self._load_checkpoint(
            row_start=row_start, row_end=row_end
        )


        if self.generation_config is None:
            logger.warning("Generation config not set. Setting default config.")
            self.set_generation_config()

        if (self.model_type == "local") and (self.llm is None):
            logger.warning("LLM not loaded. Loading default model.")
            self.load_llm()

        return self._process_df(
            formatted_column=formatted_column,
            prompt_columns=prompt_columns,
            json_schema_col=json_schema_col,
            row_start=row_start,
            row_end=row_end,
            eval_run_name=eval_run_name,
            max_prompt_length_col=max_prompt_length_col,
            max_prompt_length=max_prompt_length
        )


if __name__ == "__main__":

    from golemai.logging.log_formater import init_logger

    logger = init_logger("DEBUG")

    df = pd.read_csv(
        os.path.join(
            "/home/pim/golem_ner/golemai_package/src/golemai/nlp",
            DATA_DIR,
            DS_NAME,
        )
    )

    llm_rg = LLMRespGen(
        df=df,
        model_type="local",
        system_msg=SYSTEM_MSG_RAG,
        prompt_template=QUERY_INTRO_FEWSHOT,
        batch_size=1,
    )

    llm_rg.load_llm()

    llm_rg.generation_config = load_generation_config(
        model_id=llm_rg.model_id,
        **{
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "do_sample": False,
            "use_cache": True,
        },
    )

    llm_rg.set_prompt_template(QUERY_INTRO_NO_ANS)

    llm_rg.df = llm_rg.df.rename(columns={"question": "query"})

    resps = llm_rg.get_responses(
        prompt_columns=["query", "context"], row_start=0, row_end=3
    )
