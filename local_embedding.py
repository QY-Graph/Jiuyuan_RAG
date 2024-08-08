import json
import requests
from llama_index.core.embeddings import BaseEmbedding
from typing import List


def bge_embedding(segments):
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
    # URL = "http://localhost:8846/embedding_bge_m3"
    URL = "http://172.22.162.213:8846/embedding_bge_m3"
    r = requests.post(URL, json.dumps({'segments': segments}), headers=HEADERS)
    response = json.loads(r.content)
    return response


class LocalBgeEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def _get_text_embedding(self, text: str) -> List[float]:
        return bge_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await bge_embedding(query)

    def _get_query_embedding(self, query: str) -> List[float]:
        return bge_embedding(query)

if __name__ == "__main__":
    bge_embed_model = LocalBgeEmbedding()
    emb = bge_embed_model.get_text_embedding("test")
    print(len(emb))
