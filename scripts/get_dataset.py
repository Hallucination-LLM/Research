import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from golemai.nlp.llm_resp_gen import LLMRespGen
from golemai.nlp.hallucination_extractor import HallucinationDatasetExtractor
from golemai.nlp.prompts import SYSTEM_MSG_RAG_SHORT, QUERY_INTRO_NO_ANS
from agg_att_funcs import agg_att_weighted
import pandas as pd
import torch
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument('--model_id', type=str, default='unsloth/gemma-2-9b-it-bnb-4bit')
args = parser.parse_args()

START = args.start
END = args.end
MODEL_ID = args.model_id

EXP_NAME = 'llama2_pure'
ATT_DS_NAME = 'window_step_1__on_{examined_span_type}__all__agg_weighted'
DS_NAME = 'evaluated_df_no_bioask.parquet'

HALLU_PATH = '/net/pr2/projects/plgrid/plggllmhallu/hallu/llama2_resps/'

llm_rg = LLMRespGen(
    df=None,
    id_col='id',
    model_type='local',
    system_msg=SYSTEM_MSG_RAG_SHORT,
    prompt_template=QUERY_INTRO_NO_ANS,
)

llm_rg.load_llm(
    model_id=MODEL_ID,
    use_unsloth=False, 
    dtype=torch.bfloat16,
    token=os.environ.get('HF_TOKEN'),
    load_in_4bit=True
)

df = pd.read_parquet(os.path.join(HALLU_PATH, EXP_NAME, DS_NAME)).iloc[slice(START, END)]

hallu_ext = HallucinationDatasetExtractor(
    df=df,
    llm_rg=llm_rg,
    att_dir_path=os.path.join(HALLU_PATH, EXP_NAME, 'attentions'),
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
        'path': HALLU_PATH,
        'att_ds_name': ATT_DS_NAME.format(examined_span_type='context'),
        'start': START,
        'end': END,
    },
    agg_func=agg_att_weighted,
    use_passage_percentage=True,
    passage_perc_round=4
)

# hallu_ext.create_attension_dataset(
#     examined_span_type='query',
#     skip_first_n_tokens=8,
#     skip_last_n_tokens=2,
#     n_first_tokens=None,
#     window_size=8,
#     window_step=1,
#     valid_example_th=4,
#     exp_name=EXP_NAME,
#     saving_name_params={
#         'path': HALLU_PATH,
#         'att_ds_name': ATT_DS_NAME.format(examined_span_type='query'),
#         'start': START,
#         'end': END,
#     },
#     agg_func=agg_att_weighted,
#     use_passage_percentage=True
# )

print(f'Prepared attention dataset')