from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger
from openai import OpenAI

with open(Path(__file__).parent / "CABBAGE_agriexam.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
    
def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def CABBAGE_AgriExam_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return []

def CABBAGE_AgriExam_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc['question']
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    if doc['options']:
        options = parse_options(doc['options'])
        return f"{pre_prompt}\nQuestion: {question}\n\nOptions:\n{options}\n{post_prompt}"
    return f"{pre_prompt}\n\nQuestion: {question}{post_prompt}"

def CABBAGE_AgriExam_process_results(doc, results):
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
