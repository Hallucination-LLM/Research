import logging
from threading import Thread
from typing import Dict, List, Tuple

import requests
import torch
import transformers
from golemai.config import LOGGER_LEVEL
from golemai.enums import LocalModelAppKeys, RESTKeys
from golemai.io.file_ops import decode_stream
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)


def load_model(
    model_id,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    use_unsloth: bool = False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM | AutoTokenizer]:
    """Loads a language model and tokenizer.

    Args:
        model_id (str): The identifier of the model to load.
        max_seq_length (int): The maximum sequence length for tokenization.
        load_in_4bit (bool): Whether to load the model in 4-bit precision.
        device (str): The device to load the model on.
        dtype (torch.dtype): The data type for the model.
        use_unsloth (bool): Whether to use the unsloth library for loading the model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """


    model, tokenizer = None, None

    if use_unsloth:
        
        logger.debug(f"Using unsloth library")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
        )

        FastLanguageModel.for_inference(model)

    else:

        logger.debug(f"Using transformers library")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=kwargs.get("token"))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            load_in_4bit=load_in_4bit,
            **kwargs,
        )

        model.eval()
        model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_generation_config(model_id: str, **kwargs) -> GenerationConfig:
    """Loads the generation configuration for the model.

    Args:
        model_id (str): The identifier of the model.
        **kwargs: Additional keyword arguments.

    Returns:
        GenerationConfig: The generation configuration.
    """

    logger.debug(f"load_generation_config: {model_id = }, {kwargs = }")

    generation_config, _ = GenerationConfig.from_pretrained(
        model_id, **kwargs, return_unused_kwargs=True
    )

    return generation_config


def prepare_prompt(
    tokenizer: transformers.AutoTokenizer,
    user_input: str,
    system_input: str,
    has_system_role: bool = False,
) -> list:
    """Prepares the prompt for the model.

    Args:
        tokenizer: The tokenizer used to tokenize the input text.
        user_input (str): The user input.
        system_input (str): The system input.
        has_system_role (bool): Whether the system has a role in the conversation.

    Returns:
        list: The prepared prompt.
    """

    logger.debug(f"prepare_prompt: {has_system_role = }")

    messages = []

    if has_system_role:
        messages.append({"role": "system", "content": system_input})

    messages.append(
        {
            "role": "user",
            "content": (
                f"{system_input}{user_input}"
                if not has_system_role
                else user_input
            ),
        },
    )

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return prompt


def get_hidden_states(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    hidden_layer: int = -2,
    device: str = "cuda",
):

    logger.debug(
        f"{type(model) = }, {prompt = }, {hidden_layer = }, {device = }"
    )

    hidden_states = None
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    pred = model.forward(
        input_ids=input_ids.get("input_ids"),
        output_hidden_states=True,
    )

    del input_ids

    # pred.hidden_states = [h.detach().cpu() for h in pred.hidden_states]
    hidden_states = pred.hidden_states[hidden_layer][-1]
    del pred

    logger.debug(f"{hidden_states.shape = }, {type(hidden_states) = }")
    torch.cuda.empty_cache()

    return hidden_states


def get_text_streamer(
    tokenizer: transformers.AutoTokenizer, skip_prompt: bool = True
):

    logger.debug(f"get_text_streamer: {skip_prompt = }")

    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=skip_prompt)

    return text_streamer


def get_local_llm_response(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    text_streamer: transformers.TextIteratorStreamer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    device: str = "cuda",
    prefix_function: callable = None,
    dola: bool = False,
) -> transformers.TextIteratorStreamer:

    logger.debug(
        f"get_local_llms_response: {max_new_tokens = }, {temperature = }, {device = }, {dola = }"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = dict(
        inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
        prefix_allowed_tokens_fn=prefix_function,
    )

    if dola:
        logger.debug("Using DOLA")
        generation_kwargs["dola_layers"] = "high"
        generation_kwargs["repetition_penalty"] = 1.2

    # return generation_kwargs

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return text_streamer


def run_model_sanity_check(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    device: str,
) -> None:
    """Runs a sanity check on the model by generating text using the given
    tokenizer and model.

    Args:
        model: The model to be tested.
        tokenizer: The tokenizer used to tokenize the input text.
        device (str): The device to run the model on.
    Raises:
        RuntimeError: If the sanity check fails.
    Returns:
        None
    """

    try:

        inputs = tokenizer(
            ["what is the capitol of Poland?"], return_tensors="pt"
        ).to(device)
        outputs = model.generate(**inputs, max_new_tokens=64)
        tokenizer.batch_decode(outputs)

    except Exception as e:
        raise RuntimeError(f"Sanity check failed: {e}")

    else:
        torch.cuda.empty_cache()
        del inputs, outputs
        logger.debug(f"Sanity check passed.")


def local_llm_qa(
    system_message: str,
    prompt: str,
    host: str,
    json_schema: dict = None,
    stream: bool = True,
    timeout: int = 10,
) -> tuple[str, str]:
    """Sends a request to a local LLM model to generate a response.

    Args:
        system_message (str): System message to send to the model.
        prompt (str): User input prompt.
        host (str): Host URL for the local model.
        json_schema (dict): JSON schema for the request.
        stream (bool): Whether to stream the response.
        timeout (int): Request timeout.

    Returns:
        tuple[str, str]: Model response and model name.
    """

    logger.debug(f"local_llm_qa: {host = }, {stream = }")

    try:

        response = requests.post(
            f"{RESTKeys.HTTP}{host}",
            json={
                LocalModelAppKeys.USER_INPUT: prompt,
                LocalModelAppKeys.SYSTEM_MSG: system_message,
                LocalModelAppKeys.JSON_SCHEMA: json_schema,
            },
            stream=stream,
            timeout=timeout,
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception for prompt: {e}")
        return None, None

    if not response.ok:
        return None, None

    if not stream:
        response_json = response.json()
        model_response = response_json.get(LocalModelAppKeys.MODEL_RESPONSE)
        model_name = response_json.get(LocalModelAppKeys.MODEL_NAME)
    else:
        return decode_stream(response), None

    return model_response, model_name


def generate_response_local(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    inputs: str | list[str],
    generation_config: transformers.GenerationConfig = None,
    skip_prompt_tokens: bool = True,
    device: str = "cuda",
) -> str:
    """Generates a response using a local language model.

    Args:
        model: The model to generate the response.
        tokenizer: The tokenizer used to tokenize the input text.
        inputs (str | list[str]): The input text to generate the response.
        generation_config: The generation configuration for the model.
        device (str): The device to run the model on.

    Returns:
        str: The generated response.
    """

    logger.debug(f"generate_response_local: {len(inputs) = }, {device = }")

    try:

        inputs = tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        outputs = model.generate(
            **inputs, 
            generation_config=generation_config
        )

        inputs = inputs.to('cpu')

        if skip_prompt_tokens:

            prompt_length = inputs['input_ids'].shape[-1]

            if hasattr(outputs, "sequences"):
                outputs.sequences = outputs.sequences.detach().cpu()[:, prompt_length:]
            else:
                outputs = outputs.detach().cpu()[:, prompt_length:]
            
        # outputs = tokenizer.batch_decode(outputs.sequences if hasattr(outputs, "sequences") else outputs, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # return None
        return ["<CUDA_ERROR>"] * len(inputs)

    else:
        return outputs


def generate_response_api_old(
    prompt: str,
    system_message: str,
    host: str,
    json_schema: Dict = None,
    timeout: int = 30,
) -> str:

    logger.debug(f"generate_response_api: {host = }, {timeout = }")

    model_gen, _ = local_llm_qa(
        system_message=system_message,
        prompt=prompt,
        json_schema=json_schema,
        host=host,
        timeout=timeout,
    )

    return ["".join(model_gen)]


def generate_response_api(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:

    logger.debug(f"generate_response_api: {model_id = }")

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        **kwargs,
    )

    return completion.choices[-1].message.content

def generate_response_api_prase(
    client: OpenAI,
    model_id: str,
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:
    
    logger.debug(f"generate_response_api: {model_id = }")

    completion = client.beta.chat.completions.parse(
        model=model_id,
        messages=messages,
        **kwargs,
    )

    return completion.choices[-1].message.parsed