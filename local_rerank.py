import json
import requests

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
URL = "http://localhost:8846/get_rerank_scores_m3"

def get_rerank_scores(query, refs):
    r = requests.post(URL, json.dumps({'query': query, 'refs': refs}), headers=HEADERS)
    response = json.loads(r.content)
    return response
