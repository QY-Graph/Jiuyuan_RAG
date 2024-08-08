import os
import json
import requests
import sseclient
import random

# from openai import OpenAI
# os.environ['OPENAI_API_KEY'] = "sk-Tr2PhLcqACvVeZbd6b1e396b4c3e4204B2FfBa7a45019e9a"
# os.environ['OPENAI_BASE_URL'] = "https://yeysai.com/v1/"
# client = OpenAI()

HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
API_KEY = "znjCFoDOG4eSygs9l8imf5thk3qQb1ca"
URL = "http://127.0.0.1:8206/async/chat/api/stream/serve"

def llm(content, model_version):
    if model_version == "8b": 
        type = "plain8b"
        extra = {}
    elif model_version == "minicpm":
        type = "plainMinicpm"
        PARAMS = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": -1,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 8000,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
            "seed": random.randint(0, int(9223372036854775807 * 2 + 1)),
        }
        extra = {"param": PARAMS}

    else:
        type = "plain"
        extra = {}


    response = requests.request("POST", URL, stream=True, headers=HEADERS, data=json.dumps({'content': content, "api_key": API_KEY, "type": type, "extra": extra}))
    client = sseclient.SSEClient(response)
    for event in client.events():
        yield event.data

def llm_nostream(content, model_version):
    # # tmp: test gpt4
    # completion = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": content}]
    # )
    # return completion.choices[0].message.content
    res = ""
    for r in llm(content, model_version):
        res += json.loads(r)["text"]
    return res

def use_prompt(system_prompt_file, prompt_file, variable_dict):
    if system_prompt_file is not None:
        with open(system_prompt_file, "r") as fin:
            prompt = fin.read() + "\n\n"
    else:
        prompt = ""
    with open(prompt_file, "r") as fin:
        prompt += fin.read()
    return prompt.format(**variable_dict)