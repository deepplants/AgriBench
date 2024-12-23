from copy import deepcopy
from pathlib import Path

import os
import numpy as np
import yaml
from loguru import logger as eval_logger
from openai import OpenAI
import requests
import json
import time

NUM_SECONDS_TO_SLEEP = 5

PASSING_GRADE = 6

MULTIPLE_CHOICE_PROMPT = '\nAnswer with the option\'s letter from the given choices directly.'
OPEN_ENDED_PROMPT = ''

GPT_EVAL_MODEL_NAME = "gpt-4o-mini"#config["metadata"]["gpt_eval_model_name"]

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")


with open(Path(__file__).parent / "CABBAGE.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config_agriexam = yaml.safe_load("".join(safe_data))
    
def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string

def CABBAGE_doc_to_visual(doc):
    visual = []
    for i in range(1,6):
        if doc.get(f'image_{i}'):
            visual.append(doc[f'image_{i}'].convert("RGB"))
    return visual

def CABBAGE_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = replace_images_tokens(doc['question'])
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = ''
    if doc['question_type']=='multiple-choice':
        post_prompt = MULTIPLE_CHOICE_PROMPT
    elif doc['question_type']=='open-ended':
        post_prompt = OPEN_ENDED_PROMPT
        
    pre_prompt=''
    if doc.get('context'):
        pre_prompt = f'Context: {doc.get('context')}\n'

    if doc.get('options'):
        options = parse_options(doc['options'])
        return f"{pre_prompt}Question: {question}\n\nOptions:\n{options}\n{post_prompt}"
    return f"{pre_prompt}Question: {question}{post_prompt}"

def CABBAGE_process_results_exact_match(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = doc['answer'].strip().lower()
    pred = results[0].strip().lower()
    if pred == target:
        return {"exact_match": 1.0}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        return {"exact_match": result}
    return {"exact_match": 0.0}

rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))

def get_eval(content: str, max_tokens: int, retries: int = 5):
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise agronomy assistant for checking the quality of the answer.",
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def parse_score(review):
    score = review.split("\n")[0]
    score = score.replace(",", " ")
    try:
        return float(score)
    except ValueError:
        eval_logger.debug(f"Score not parsed: {review}. Returning -1")
        return -1

def CABBAGE_process_results_gpt_eval(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    question = doc.get("question", "")
    ans1 = doc.get("answer", "")
    ans2 = results[0] if results else ""
    
    if doc['question_type']=='multiple-choice':
        exact_match_result = CABBAGE_process_results_exact_match(doc, results)
        review_dict = {'gpt_eval_score': {"question": question, "ans1": ans1, "ans2": ans2, "difficulty": doc.get('options_difficulty'), "review": '',  "score": exact_match_result['exact_match']*10, "eval_model": '', "content": ''},
                       'passing_adjusted_score': exact_match_result['exact_match']}
        return review_dict
    
    try:
        question = doc.get("question", "")
        ans1 = doc.get("answer", "")
        role1= "Expert"
        ans2 = results[0] if results else ""
        role2 = rule_dict.get("role", "user")
        
        captions = doc.get("caption", [])
        # TODO add the context label to the dataset (docs)
        #context = config['lmms_eval_specific_kwargs']['default']['context_prompt'].format(book_title=book_title, chapter_title=chapter_title)
        context=''
        if doc.get('category'):
            context += f'Category: {doc.get('category')}\n'
        if doc.get('context'):
            context += f'Context: {doc.get('context')}\n'
        
        prompt = rule_dict.get("prompt", "")
        content = f"[Context]\n{context}\n" f"[Question]\n{question}\n\n" f"[{role1}]\n{ans1}\n\n[End of {role1}]\n\n" f"[{role2}]\n{ans2}\n\n[End of {role2}]\n\n" f"[System]\n{prompt}\n\n"
        review, model_name = get_eval(content, 1024)
        score = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        score = -1
    if score >= PASSING_GRADE:
        value = 1.0
    else:
        value = 0.0
        
    review_dict = {'gpt_eval_score': {"question": question, "ans1": ans1, "ans2": ans2, "difficulty": doc.get('options_difficulty'), "review": review, "score": score, "eval_model": model_name, "content": content},
                   'passing_adjusted_score' : value}

    return review_dict

def gpt_eval_aggregation(results):
    try:
        scores = []
        for result in results:
            if result["score"] == -1:
                continue
            scores.append(result["score"])

        stats = np.asarray(scores).mean(0).tolist()
        stats = round(stats, 3)
        return stats/10
    except Exception as e:
        eval_logger.info(f"Error in aggregation: {e}")
        return None
