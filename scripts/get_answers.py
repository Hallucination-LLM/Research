import os
import pandas as pd
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_dola", action='store_true', default=False)
    parser.add_argument("--fewshot", action='store_true', default=False)
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    DOLA = args.use_dola
    FEWSHOT = args.fewshot
    START = args.start
    END = args.end
    DEVICE_NUM = args.device_num
    CHECKPOINT_FILE = args.checkpoint_file
    EXP_NAME = args.exp_name

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE_NUM}"

    from golemai.nlp.llm_resp_gen import LLMRespGen
    from golemai.nlp.prompts import SYSTEM_MSG_RAG, QUERY_INTRO_NO_ANS, QUERY_INTRO_FEWSHOT

    DATA_DIR = 'data'
    DS_NAME = 'test_gemma_resp.csv'

    df = pd.read_csv(os.path.join("..", DATA_DIR, DS_NAME))

    llm_rg = LLMRespGen(
        df=df,
        model_type='local',
        system_msg=SYSTEM_MSG_RAG,
        prompt_template=QUERY_INTRO_NO_ANS if not FEWSHOT else QUERY_INTRO_FEWSHOT,
        batch_size=1,
        device_num=DEVICE_NUM
    )

    if CHECKPOINT_FILE is not None:

        llm_rg.configure_checkpoint(
            checkpoint_path=os.path.join("checkpoints", CHECKPOINT_FILE)
        )

    llm_rg.load_llm(use_unsloth=False)

    llm_rg.set_generation_config(
        model_id=llm_rg.model_id,
        **{
            "max_new_tokens": 200,
            "temperature": 0.0,
            # "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_attentions": True,
            "output_hidden_states": True,
        },
        dola_layers="high" if DOLA else None,
        repetition_penalty=1.2 if DOLA else 1.0
    )

    llm_rg.df = llm_rg.df.rename(columns={'question': 'query'})

    resps = llm_rg.get_responses(
        eval_run_name=EXP_NAME,
        prompt_columns=['query', 'context'],
        row_start=START,
        row_end=END
    )

# for i in range(0, 1):
#     print(resps['model_responses'][i].split('\nmodel\n')[-1].strip())
#     print(df.iloc[i]['gemma-2-9b-it-bnb-4bit'].strip())
#     print('\n\n')