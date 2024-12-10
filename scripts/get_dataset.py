import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from golemai.nlp.llm_resp_gen import LLMRespGen
from golemai.nlp.hallucination_extractor import HallucinationDatasetExtractor
from golemai.nlp.prompts import SYSTEM_MSG_RAG_SHORT, QUERY_INTRO_NO_ANS
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

START = args.start
END = args.end

EXP_NAME = 'gemma_att'
ATT_DS_NAME = 'window_step_1__on_{examined_span_type}__all'
DS_NAME = 'all_evaluated_df.parquet'

llm_rg = LLMRespGen(
    df=None,
    id_col='id',
    model_type='local',
    system_msg=SYSTEM_MSG_RAG_SHORT,
    prompt_template=QUERY_INTRO_NO_ANS,
)

llm_rg.load_llm(use_unsloth=False, dtype=torch.bfloat16)

df = pd.read_parquet(os.path.join(EXP_NAME, DS_NAME)).iloc[slice(START, END)]

hallu_ext = HallucinationDatasetExtractor(
    df=df,
    llm_rg=llm_rg,
    att_dir_path=os.path.join(EXP_NAME, 'attentions'),
)

df_hallu = hallu_ext.prepare_hallucinated_df_info(exp_name=EXP_NAME)

print(f'Prepared hallu df')

hallu_ext.create_attension_dataset(
    examined_span_type='context',
    skip_first_n_tokens=8,
    skip_last_n_tokens=2,
    n_first_tokens=None,
    window_size=8,
    window_step=1,
    valid_example_th=4,
    exp_name=EXP_NAME,
    saving_name_params={
        'att_ds_name': ATT_DS_NAME.format(examined_span_type='context'),
        'start': START,
        'end': END,
    }
)

hallu_ext.create_attension_dataset(
    examined_span_type='query',
    skip_first_n_tokens=8,
    skip_last_n_tokens=2,
    n_first_tokens=None,
    window_size=8,
    window_step=1,
    valid_example_th=4,
    exp_name=EXP_NAME,
    saving_name_params={
        'att_ds_name': ATT_DS_NAME.format(examined_span_type='query'),
        'start': START,
        'end': END,
    }
)

print(f'Prepared attention dataset')