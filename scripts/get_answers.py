import os
import pandas as pd
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dola", action='store_true', default=False)
    parser.add_argument("--fewshot", action='store_true', default=False)
    parser.add_argument("--device_num", type=str, default='auto')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--return_dict_in_gen", action='store_true', default=False)

    args = parser.parse_args()

    DOLA = args.use_dola
    FEWSHOT = args.fewshot
    START = args.start
    END = args.end
    DEVICE_NUM = args.device_num
    CHECKPOINT_FILE = args.checkpoint_file
    EXP_NAME = args.exp_name
    RETURN_DICT_IN_GEN = args.return_dict_in_gen

    MODEL_ID = 'gemma-2-9b-it-bnb-4bit'

    if DEVICE_NUM != 'auto':
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE_NUM}"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from dotenv import load_dotenv
    from golemai.nlp.llm_resp_gen import LLMRespGen
    from golemai.nlp.prompts import SYSTEM_MSG_RAG_SHORT, QUERY_INTRO_NO_ANS, QUERY_INTRO_FEWSHOT, PROMPT_QA
    from golemai.nlp.llm_evaluator import LLMEvaluator
    from datetime import datetime
    import torch

    load_dotenv()

    DATA_DIR = 'data'
    DS_NAME = 'data_filtered_sample_1500.parquet'

    df = pd.read_parquet(os.path.join("..", DATA_DIR, DS_NAME)).reset_index(drop=True)

    llm_rg = LLMRespGen(
        df=df,
        id_col='id',
        model_type='local',
        system_msg=SYSTEM_MSG_RAG_SHORT,
        prompt_template=QUERY_INTRO_NO_ANS if not FEWSHOT else QUERY_INTRO_FEWSHOT,
        batch_size=1,
        device_num=DEVICE_NUM
    )

    if CHECKPOINT_FILE is not None:

        llm_rg.configure_checkpoint(
            checkpoint_path=os.path.join("checkpoints", CHECKPOINT_FILE)
        )

    llm_rg.load_llm(use_unsloth=False, dtype=torch.bfloat16)

    llm_rg.set_generation_config(
        model_id=llm_rg.model_id,
        **{
            "max_new_tokens": 200,
            # "temperature": 0.0,
            "do_sample": False,
            "use_cache": True,
            # "cache_implementation": None,
            "return_dict_in_generate": RETURN_DICT_IN_GEN,
            "output_attentions": True,
            "output_hidden_states": False,
        },
        dola_layers="high" if DOLA else None,
        repetition_penalty=1.2 if DOLA else 1.0
    )

    llm_rg.configure_att_hidden_config(
        prompt_offset=8,
        take_only_generated=True
    )

    llm_rg.df = llm_rg.df.rename(columns={'question': 'query'})

    resps = llm_rg.get_responses(
        eval_run_name=EXP_NAME,
        prompt_columns=['query', 'context'],
        row_start=START,
        row_end=END,
        max_prompt_length_col='context_length',
        max_prompt_length=3996
    )

    resps = resps['model_responses']

    api_key = os.getenv("OPENAI_API_KEY")

    evaluator = LLMEvaluator(
        id_col='id',
        model_type="openai",
        api_url="https://api.openai.com/v1/",
        api_key=api_key,
        system_msg="You are a helpful assistant.",
        prompt_template=PROMPT_QA,
        has_system_role=True,
        use_pydantic=False,
    ).set_generation_config(
        model_id="gpt-4o",
    )


    evaluator.evaluate(
        df=df,
        exp_name=EXP_NAME if EXP_NAME is not None else f'{MODEL_ID}_eval_{datetime.now().strftime("%Y-%m-%d_%H-%M")}',
        responses=resps,
    )