import argparse
import json
import os
import openai
import tiktoken
import torch
import pandas as pd
from prompts import DATA_RESPONSE_NAMES, EVAL_PROMPT_BEFORE, EVAL_PROMPT_AFTER, DATA_RESPONSE_NAMES_GT
from evaluation_schema import DataType

DEBUG = False
DEBUG_LIMIT = 5

# Function to load data from various formats
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
            'net_response': entry['answer']  # Assuming 'answer' is the proposed summary
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
            'net_response': entry['answer'],  # Assuming 'answer' is the ground truth answer
            'data_index': idx
        }
    return list_data_dict

# Function to evaluate responses using GPT-4o
def evaluate_response(document, gt_response, response, tokenizer, data_type=DataType.SUMMARIZATION):
    prompt = f"{EVAL_PROMPT_BEFORE[data_type]}\n\n#Document#: {document}\n\n#Ground Truth {DATA_RESPONSE_NAMES_GT[data_type]}#: {gt_response}\n\n#Proposed {DATA_RESPONSE_NAMES[data_type]}#: {response}\n\n{EVAL_PROMPT_AFTER[data_type]}"
    
    # Calculate input token usage
    input_token_usage = len(tokenizer.encode(prompt))

    response = openai.chat.completions.create(
        model='gpt-4o-2024-05-13',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content
    
    # Calculate output token usage
    output_token_usage = len(tokenizer.encode(text))

    if DEBUG:
        print('-------------------')
        print(prompt)
        print('\n' + text + '\n')
        print('-------------------', flush=True)

    problematic_spans = []
    if "Problematic Spans: " in text:
        problematic_spans = text.split('Problematic Spans: ')[1]
        if '**' in problematic_spans:
            problematic_spans = problematic_spans.split('**')[0].strip()
        # problematic_spans is in python list of string format, extract the list
        try:
            problematic_spans = eval(problematic_spans)
        except Exception as e:
            print("Error in parsing problematic spans:", problematic_spans, e)
            problematic_spans = problematic_spans[1:-1].split(', ')

        if DEBUG:
            print(problematic_spans)

    if "Conclusion: " in text:
        dec = text.split('Conclusion: ')[1]
        if '**' in dec:
            dec = dec.split('**')[0]
        if DEBUG:
            print(dec)
        decision = "True" in dec
    else:
        decision = None
    
    # Calculate cost
    cost = (input_token_usage / 1_000_000 * 5) + (output_token_usage / 1_000_000 * 15)
    
    return decision, text, problematic_spans, cost

def main(hyp_path, ref_path, output_path, limit=None, model='gpt-4o-2024-05-13'):
    # Load data files
    gold_data = load_data(ref_path)
    data_type = DataType.SUMMARIZATION if DataType.SUMMARIZATION in hyp_path else DataType.QA
    
    if limit is not None:
        gold_data = gold_data[:limit]

    if data_type == DataType.SUMMARIZATION:
        gold_data_dict = load_summarization(gold_data)
    else:
        gold_data_dict = load_q_and_a(gold_data)

    # Load responses
    response_data = load_data(hyp_path)
    responses = [item['answer'] for item in response_data]

    if limit is not None:
        responses = responses[:limit]
        
    if DEBUG:
        responses = responses[:DEBUG_LIMIT]

    # Initialize OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("o200k_base")

    done_dict = {}
    if os.path.exists(output_path):
        print("Trying to resume from existing output file.")
        with open(output_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                data = json.loads(line)
                done_dict[data['index']] = data

    # Open output file
    with open(output_path, 'w') as fw:
        results = []
        total_cost = 0
        corr = 0
        total = 0

        # Evaluate each pair of responses
        for idx in range(len(responses)):
            response = responses[idx]
            if idx not in gold_data_dict:
                continue
            document = gold_data_dict[idx]['context']
            gt_response = gold_data_dict[idx]['net_response']

            if idx in done_dict:
                fw.write(json.dumps(done_dict[idx]) + '\n')
                continue
            decision, gpt4_explanation, problematic_spans, cost = evaluate_response(
                document, gt_response, response, tokenizer, data_type=data_type
            )
            results.append({
                'index': idx, 
                'document': document.strip(), 
                'ground_truth': gt_response.strip(), 
                'response': response, 
                'decision': decision, 
                'gpt4_explanation': gpt4_explanation, 
                'problematic_spans': problematic_spans, 
                'cost': cost
            })
            fw.write(json.dumps(results[-1]) + '\n')
            fw.flush()

            # Accumulate total cost
            total_cost += cost

            # Accuracy
            total += 1
            if decision:
                corr += 1

        # Print total cost and accuracy
        print(f"Total cost: ${total_cost:.9f}")
        print(f"Accuracy: {corr / total:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate faithfulness of summaries and Q&A answers using GPT-4o.")
    parser.add_argument('--hyp', type=str, required=True, help='Path to the hypothesis file (CSV, JSONL, Parquet)')
    parser.add_argument('--ref', type=str, required=True, help='Path to the reference file (CSV, JSONL, Parquet)')
    parser.add_argument('--out', type=str, required=True, help='Path to the output file')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13', help='Model name to use for evaluation')

    args = parser.parse_args()
    main(args.hyp, args.ref, args.out, args.limit, args.model)

#Usage OPENAI_API_KEY=[API_KEY] python gbt_evaluator.py --hyp [HYP_PATH] --ref [REF_PATH] --out [OUTPUT_PATH] --limit [LIMIT] --model [MODEL_NAME]