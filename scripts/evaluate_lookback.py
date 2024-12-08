import os
import argparse
import json
import pandas as pd
from golemai.nlp.prompts import PROMPT_QA
from golemai.nlp.llm_evaluator import LLMEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--resps_file", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    DATA = args.data_file
    RESPS = args.resps_file
    EXP_NAME = args.exp_name

    df = pd.read_parquet(DATA)

    with open(RESPS, "r") as f:
        resps = json.load(f)

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

    os.makedirs(EXP_NAME)
    os.makedirs(f"{EXP_NAME}/checkpoints")

    df = evaluator.evaluate(
        df=df,
        exp_name=EXP_NAME,
        responses=resps,
        row_start=0,
        row_end=None,
    )
