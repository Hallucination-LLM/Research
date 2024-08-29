import transformers
import torch
import logging
from unsloth import FastLanguageModel
from config import LOGGER_LEVEL
from transformers import TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

def load_model(
    model_id,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    device: str = 'cuda:0',
    use_unsloth: bool = False
    ):
    model, tokenizer = None, None

    if use_unsloth:
        logger.info(f"Using unsloth library")
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        device_map={"": device},
        load_in_4bit=load_in_4bit
        )

        FastLanguageModel.for_inference(model)

    else:
        logger.info(f"Using transformers library")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if load_in_4bit else torch.float32,
            )


        model.eval()


    return model, tokenizer
        
def prepare_prompt(
        tokenizer,
        user_input: str, 
        system_input: str,
        has_system_role: bool = False) -> list:
    
    messages = []
    
    if has_system_role:
        messages.append({"role": "system", "content": system_input})

    messages = [
        {
            "role": "user", 
            "content": f"{system_input}\n\n{user_input}" 
                if not has_system_role 
                else user_input
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    return prompt

def get_hidden_states(
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        prompt: str,
        hidden_layer: int = -2,
        device: str = 'cuda'):
    
    logger.info(f"{type(model) = }, {prompt = }, {hidden_layer = }, {device = }")

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

    logger.info(f"{hidden_states.shape = }, {type(hidden_states) = }")
    torch.cuda.empty_cache()

    return hidden_states

def get_text_streamer(
    tokenizer: transformers.AutoTokenizer,
    skip_prompt: bool = True):

    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=skip_prompt)

    return text_streamer

def get_local_llms_response(
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        text_streamer: transformers.TextIteratorStreamer,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        device: str = 'cuda',
        prefix_function: callable = None,
        dola: bool = False,
        ):

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = dict(
        inputs, 
        streamer=text_streamer, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        use_cache=True,
        prefix_allowed_tokens_fn=prefix_function
    )

    if dola:
        logger.info("Using DOLA")
        generation_kwargs["dola_layers"] = 'high'
        generation_kwargs["repetition_penalty"] = 1.2

    # return generation_kwargs

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return text_streamer