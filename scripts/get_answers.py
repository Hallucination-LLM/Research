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
    parser.add_argument("--task_type", type=str, default='qa')

    args = parser.parse_args()

    DOLA = args.use_dola
    FEWSHOT = args.fewshot
    START = args.start
    END = args.end
    DEVICE_NUM = args.device_num
    CHECKPOINT_FILE = args.checkpoint_file
    EXP_NAME = args.exp_name
    RETURN_DICT_IN_GEN = args.return_dict_in_gen
    TASK_TYPE = args.task_type

    MODEL_ID = 'gemma-2-9b-it-bnb-4bit'

    if DEVICE_NUM != 'auto':
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE_NUM}"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from dotenv import load_dotenv
    from golemai.nlp.llm_resp_gen import LLMRespGen
    from golemai.nlp.hallucination_extractor import HallucinationDatasetExtractor
    from golemai.nlp.prompts import SYSTEM_MSG_RAG_SHORT, QUERY_INTRO_NO_ANS, QUERY_INTRO_FEWSHOT, PROMPT_QA, PROMPT_SUMMARIZATION
    from golemai.nlp.llm_evaluator import LLMEvaluator
    from datetime import datetime
    import torch

    load_dotenv()

    REPO_DIR = 'Research'
    DATA_DIR = 'data'
    DS_NAME = 'cnndm.parquet'

    TASKS = {
        'qa': PROMPT_QA,
        'summ': PROMPT_SUMMARIZATION
    }

    df = pd.read_parquet(os.path.join(REPO_DIR, DATA_DIR, DS_NAME)).reset_index(drop=True)

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
            checkpoint_path=os.path.join(EXP_NAME, "checkpoints", CHECKPOINT_FILE),
            checkpoint_freq=2
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
            "skip_prompt_tokens": True,
            "skip_special_tokens": True,
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
        max_prompt_length=3896
    )

    resps = resps['model_responses']
    print(resps)

    api_key = os.getenv("OPENAI_API_KEY")

    evaluator = LLMEvaluator(
        id_col='id',
        model_type="openai",
        api_url="https://api.openai.com/v1/",
        api_key=api_key,
        system_msg="You are a helpful assistant.",
        prompt_template=TASKS.get(TASK_TYPE, 'qa') ,
        has_system_role=True,
        use_pydantic=False,
    ).set_generation_config(
        model_id="gpt-4o",
    )


    df = evaluator.evaluate(
        df=df,
        exp_name=EXP_NAME if EXP_NAME is not None else f'{MODEL_ID}_eval_{datetime.now().strftime("%Y-%m-%d_%H-%M")}',
        row_start=START,
        row_end=END,
        responses=resps,
        checkpoint_file=f'evaluated_{START}_{END}.json'
    )

    print(df.head())

    hallu_ext = HallucinationDatasetExtractor(
        df=df,
        llm_rg=llm_rg,
        att_dir_path=os.path.join(EXP_NAME, 'attentions'),
    )

    df = hallu_ext.prepare_hallucinated_df_info(exp_name=EXP_NAME)
    print(df.head())

    hallu_df = hallu_ext.create_attension_dataset(
        examined_span_type='context',
        skip_first_n_tokens=8,
        n_first_tokens=None,
        window_size=8,
        exp_name=EXP_NAME,
        save_name=f'attension_{START}_{END}_df'
    )

    print(f'Prepared attention dataset')