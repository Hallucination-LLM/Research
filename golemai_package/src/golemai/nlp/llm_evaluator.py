import json
import logging
import os
from collections import namedtuple

import pandas as pd
import tiktoken
from golemai.nlp.evaluation_schema import (
    DataType,
    EvaluationResult,
    IsAnswerCorrect,
)
from golemai.nlp.llm_resp_gen import LLMRespGen
from golemai.nlp.prompts import PROMPT_QA
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

EvaluationResultTuple = namedtuple('EvaluationResult', ['decision', 'explanation', 'problematic_spans'])


class LLMEvaluator(LLMRespGen):
    def __init__(self, result_path='results.json', use_pydantic=True, **kwargs):
        super().__init__(
            **kwargs
        )
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.result_path = result_path
        self.use_pydantic = use_pydantic

    def evaluate_response(self, document, gt_response, response):
        prompt = self.prepare_prompt(
            document=document,
            gt_response=gt_response,
            response=response,
        )

        input_tokens = len(self.tokenizer.encode(prompt))

        if self.use_pydantic:
            result = self._generate_llm_response(
                inputs=prompt,
                pydantic_model=EvaluationResult,
            )
        else:
            result_text = self._generate_llm_response(inputs=prompt)
            result = self._parse_raw_response(result_text)

        output_tokens = len(self.tokenizer.encode(result.explanation))

        cost = self._calculate_cost(input_tokens, output_tokens)

        logger.debug('-------------------')
        logger.debug(prompt)
        logger.debug('\n' + result.explanation + '\n')
        logger.debug('-------------------')
        logger.debug("Problematic spans: %s", result.problematic_spans)

        return result, cost, prompt
    
    def evaluate(self, df, exp_name, responses, limit=None):
        df = df.loc[df['id'].isin(list(responses.keys()))]
        df[exp_name] = df['id'].map(responses)

        df = df.rename(columns={'query': 'question'})


        responses = df[['id', exp_name]]
        responses.columns = ['id', 'answer']

        self.result_path = f'{exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json'
        results, total_cost, accuracy = self.evaluate_from_dfs(
            df, responses
        )
        
        print(f"Total Cost: {total_cost}")
        print(f"Accuracy: {accuracy}")

        return results, total_cost, accuracy

    def evaluate_from_dfs(self, gt_df, response_df, limit=None):

        gt_data = self._prepare_gt_data(gt_df, limit)
        done_results = self._load_done_results()
        total_cost, correct_count, results = 0, 0, []

        with open(self.result_path, 'a') as fw:

            for idx in tqdm(response_df[self.id_col].head(limit) if limit else response_df[self.id_col],
                            desc="Evaluating responses"):
                
                if idx not in gt_data or idx in done_results:
                    results.append(done_results.get(idx, {}))
                    continue

                response = response_df.loc[response_df[self.id_col] == idx, 'answer'].values[0]

                if response in ["<SKIPPED>", "<CUDA_ERROR>"]:
                    results.append({})
                    continue

                document = gt_data[idx]['context']
                gt_response = gt_data[idx]['net_response']

                result, cost, prompt = self.evaluate_response(document, gt_response, response, )

                results.append(self._record_result(idx, document, gt_response, response, result, cost, prompt))
                fw.write(json.dumps(results[-1], ensure_ascii=False) + '\n')
                fw.flush()

                total_cost += cost

                if self.use_pydantic:
                    correct_count += result.decision == IsAnswerCorrect.TRUE
                else:
                    correct_count += result.decision

        accuracy = correct_count / len(results) if len(results) else 0

        logger.info(f"Total cost: ${total_cost:.9f}")
        logger.info(f"Accuracy: {accuracy:.3f}")

        return results, total_cost, accuracy

    def _parse_raw_response(self, response_text):
        explanation = ""
        problematic_spans = []
        decision = None

        # Extract explanation
        explanation = response_text.split("Problematic Spans:")[0].strip()

        if '**' in explanation:
            explanation = explanation.split('**')[0].strip()

        # Extract problematic spans
        if "Problematic Spans:" in response_text:
            spans_text = response_text.split("Problematic Spans: ")[1].split("Conclusion:")[0].strip()
            if '**' in spans_text:
                spans_text = spans_text.split('**')[0].strip()
            # Safely parse the list of spans
            try:
                problematic_spans = [span.strip() for span in spans_text.strip('[]').split(',')]
            except Exception as e:
                logger.error(f"Error parsing problematic spans: {e}")

        # Extract conclusion
        if "Conclusion:" in response_text:
            conclusion_text = response_text.split("Conclusion: ")[1].strip()
            if '**' in conclusion_text:
                conclusion_text = conclusion_text.split('**')[0].strip()
            decision = "True" in conclusion_text

        return EvaluationResultTuple(decision=decision, explanation=explanation, problematic_spans=problematic_spans)

    def _calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens / 1_000_000 * 5) + (output_tokens / 1_000_000 * 15)

    def _prepare_gt_data(self, gt_df, limit):
        data = gt_df.head(limit) if limit else gt_df

        return {
            entry[self.id_col]: {
                'context': f"#Document#: {entry['context']}\n#Question#: {entry['question'].capitalize()}",
                'net_response': entry['answer']
            }
            for _, entry in data.iterrows()
        }

    def _load_done_results(self):
        done_dict = {}
        if os.path.exists(self.result_path):
            with open(self.result_path, 'r') as fr:
                for line in fr:
                    data = json.loads(line)
                    done_dict[data[self.id_col]] = data
        return done_dict

    def _record_result(self, idx, document, gt_response, response, result, cost, prompt):
        return {
            self.id_col: idx,
            'document': document.strip(),
            'ground_truth': gt_response.strip(),
            'response': response,
            'decision': result.decision,
            'gpt4_explanation': result.explanation,
            'problematic_spans': result.problematic_spans if not self.use_pydantic else [span.span for span in
                                                                                         result.problematic_spans],
            'cost': cost,
            'prompt': prompt
        }


if __name__ == '__main__':

    from golemai.logging.log_formater import init_logger
    from dotenv import load_dotenv

    load_dotenv()
    logger = init_logger('ERROR')

    api_key = os.getenv("OPENAI_API_KEY")

    evaluator = LLMEvaluator(
        model_type="openai",
        api_url="https://api.openai.com/v1/",
        api_key=api_key,
        system_msg="You are a helpful assistant.",
        prompt_template=PROMPT_QA,
        has_system_role=True,
        use_pydantic=False,
        result_path = 'sample_dataset_eval_results.json'
    ).set_generation_config(
        model_id="gpt-4o-mini-2024-07-18",
    )

    df = pd.read_parquet('test_gemma_resp_dola.parquet')

    df_eval = df[df['dataset'] == 'POLQA']

    COLUMNS_TO_EVAL = [
        'gemma-2-9b-it-bnb-4bit',
        'gemma-2-9b-it-bnb-4bit-dola',
        'gemma-2-9b-it-bnb-4bit-few-shot-dola',
        'gemma-2-9b-it-bnb-4bit-few-shot'
    ]

    for column in COLUMNS_TO_EVAL:
        responses = df_eval.reset_index()[['index', column]]
        responses.columns = ['index', 'answer']

        evaluator.result_path = f'{column}.json'
        results, total_cost, accuracy = evaluator.evaluate_from_dfs(
            df_eval, responses, limit=202
        )
        print(f"Model: {column}")
        # print(f"Results: {results}")
        print(f"Total Cost: {total_cost}")
        print(f"Accuracy: {accuracy}")
