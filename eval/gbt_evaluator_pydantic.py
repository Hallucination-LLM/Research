import argparse
import json
import os
from enum import Enum
from typing import List
import openai
import tiktoken
import pandas as pd
from pydantic import BaseModel, Field
from prompts import DATA_RESPONSE_NAMES, EVAL_PROMPT_BEFORE, EVAL_PROMPT_AFTER, DATA_RESPONSE_NAMES_GT
from evaluation_schema import DataType, IsAnswerCorrect, EvaluationResult, OutputData

DEBUG = True
DEBUG_LIMIT = 5

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path).to_dict(orient='records')
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path).to_dict(orient='records')
    else:
        raise ValueError("Unsupported file format. Use CSV, JSONL, or Parquet.")

def load_summarization(data):
    if DEBUG:
        data = data[:DEBUG_LIMIT]
    list_data_dict = {}
    for idx, entry in enumerate(data):
        context = "#Document#: " + entry['context']
        list_data_dict[idx] = {
            'context': context,
            'data_index': idx,
            'net_response': entry['answer']
        }
    return list_data_dict

def load_q_and_a(data):
    list_data_dict = {}
    for idx, entry in enumerate(data):
        question = entry['question']
        question = question[0].upper() + question[1:] + '?' if question[-1] != '?' else question
        context = "#Document#: " + entry['context'] + f"\n#Question#: {question}"
        list_data_dict[idx] = {
            'context': context,
            'response': f"\n#Answer#: ",
            'net_response': entry['answer'],
            'data_index': idx
        }
    return list_data_dict

def evaluate_response(document, gt_response, response, tokenizer, data_type=DataType.SUMMARIZATION, model='gpt-4o-2024-05-13'):
    prompt = f"{EVAL_PROMPT_BEFORE[data_type]}\n\n#Document#: {document}\n\n#Ground Truth {DATA_RESPONSE_NAMES_GT[data_type]}#: {gt_response}\n\n#Proposed {DATA_RESPONSE_NAMES[data_type]}#: {response}\n\n{EVAL_PROMPT_AFTER[data_type]}"

    input_token_usage = len(tokenizer.encode(prompt))
    
    completion = openai.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format=EvaluationResult
    )
    
    result = completion.choices[0].message.parsed
    output_token_usage = len(tokenizer.encode(result.explanation))
    
    if DEBUG:
        print('-------------------')
        print(prompt)
        print('\n' + result.explanation + '\n')
        print('-------------------', flush=True)
        print("Problematic spans:", [span.span for span in result.problematic_spans])
    
    # Calculate cost
    cost = (input_token_usage / 1_000_000 * 5) + (output_token_usage / 1_000_000 * 15)
    
    return result, cost, prompt

def main(hyp_path, ref_path, output_path, limit=None, model='gpt-4o-2024-05-13'):
    # Load data files
    gold_data = load_data(ref_path)
    data_type = DataType.QA

    if limit is not None:
        gold_data = gold_data[:limit]

    gold_data_dict = load_q_and_a(gold_data) if data_type == DataType.QA else load_summarization(gold_data)

    # Load responses
    response_data = load_data(hyp_path)
    responses = [item['answer'] for item in response_data]

    if limit is not None:
        responses = responses[:limit]
        
    if DEBUG:
        responses = responses[:DEBUG_LIMIT]

    # Initialize OpenAI API key and tokenizer
    openai.api_key = os.getenv("OPENAI_API_KEY")
    tokenizer = tiktoken.get_encoding("o200k_base")

    # Load existing results
    done_dict = {}
    if os.path.exists(output_path):
        print("Trying to resume from existing output file.")
        with open(output_path, 'r') as fr:
            for line in fr:
                data = json.loads(line)
                done_dict[data['index']] = data

    # Process evaluations
    with open(output_path, 'w') as fw:
        total_cost = 0
        corr = 0
        total = 0

        for idx in range(len(responses)):
            if idx not in gold_data_dict:
                continue

            if idx in done_dict:
                fw.write(json.dumps(done_dict[idx]) + '\n')
                continue

            response = responses[idx]
            document = gold_data_dict[idx]['context']
            gt_response = gold_data_dict[idx]['net_response']

            result, cost, prompt = evaluate_response(
                document, gt_response, response, tokenizer, data_type=data_type, model=model
            )

            output_data = OutputData(
                index=idx,
                document=document.strip(),
                ground_truth=gt_response.strip(),
                response=response,
                decision=result.decision,
                gpt4_explanation=result.explanation,
                problematic_spans=[span.span for span in result.problematic_spans],
                cost=cost,
                prompt=prompt
            )

            fw.write(output_data.model_dump_json() + '\n')
            fw.flush()

            total_cost += cost
            total += 1
            if result.decision == IsAnswerCorrect.NO:
                corr += 1

        print(f"Total cost: ${total_cost:.9f}")
        print(f"Accuracy: {corr / total:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate faithfulness of summaries and Q&A answers using GPT-4o.")
    parser.add_argument('--hyp', type=str, required=True, help='Path to the hypothesis file (CSV, JSONL, Parquet)')
    parser.add_argument('--ref', type=str, required=True, help='Path to the reference file (CSV, JSONL, Parquet)')
    parser.add_argument('--out', type=str, required=True, help='Path to the output file in JSONL format')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    parser.add_argument('--model', type=str, default='gpt-4o-mini-2024-07-18', help='Model to use for evaluation')

    args = parser.parse_args()
    main(args.hyp, args.ref, args.out, args.limit, args.model)

# OPENAI_API_KEY=[KEY] python eval/gbt_evaluator_pydantic.py --hyp data/response.csv --ref data/golden_data.csv --out data/eval_pydantic.jsonl --limit 2