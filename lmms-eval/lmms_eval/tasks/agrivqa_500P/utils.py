import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import openai
import requests
import yaml
from loguru import logger as eval_logger
from openai import OpenAI

NUM_SECONDS_TO_SLEEP = 5

rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))

with open(Path(__file__).parent / "agrivqa_500P.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


def get_eval(content: str, max_tokens: int, retries: int = 5):
    global headers

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

    if API_TYPE == "azure":
        payload.pop("model")

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

########################################



# def agrivqa_500P_doc_to_visual(doc):
#     return [doc["image"].convert("RGB")]


def agrivqa_500P_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    book_title = eval(doc['metadata']).get('book_title','')
    chapter_title = eval(doc['metadata']).get('chapter_title','')
    question = doc['question']
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    context_prompt = lmms_eval_specific_kwargs.get("context_prompt", "").format(book_title=book_title, chapter_title=chapter_title)
    return f"{pre_prompt}\n\nContext: {context_prompt}\n\nQuestion: {question}{post_prompt}"


def agrivqa_500P_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    try:
        book_title = eval(doc['metadata']).get('book_title','')
        chapter_title = eval(doc['metadata']).get('chapter_title','')
        
        question = doc.get("question", "")
        ans1 = doc.get("answer", "")
        role1= "Expert"
        ans2 = result[0] if result else ""
        role2 = rule_dict.get("role", "user")
        
        captions = doc.get("caption", [])
        # TODO add the context label to the dataset (docs)
        context = config['lmms_eval_specific_kwargs']['default']['context_prompt'].format(book_title=book_title, chapter_title=chapter_title)
        prompt = rule_dict.get("prompt", "")
        content = f"[Context]\n{context}\n\n" f"[Question]\n{question}\n\n" f"[{role1}]\n{ans1}\n\n[End of {role1}]\n\n" f"[{role2}]\n{ans2}\n\n[End of {role2}]\n\n" f"[System]\n{prompt}\n\n"
        review, model_name = get_eval(content, 1024)
        score = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        score = -1

    review_dict = {'gpt_eval_agrivqa_500P': {"question": question, "ans1": ans1, "ans2": ans2, "difficulty": doc['topic_difficulty'], "review": review, "score": score, "eval_model": model_name, "content": content} }

    return review_dict


def agrivqa_500P_aggregation(results):
    try:
        scores = []
        for result in results:
            if result["score"] == -1:
                continue
            scores.append(result["score"])

        stats = np.asarray(scores).mean(0).tolist()
        stats = round(stats, 3)
        return stats*10
    except Exception as e:
        eval_logger.info(f"Error in agrivqa_500P_aggregation: {e}")
        return None
