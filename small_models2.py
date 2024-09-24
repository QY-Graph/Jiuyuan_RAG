import os
from typing import List
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request
# from FlagEmbedding import BGEM3FlagModel, FlagReranker, FlagModel
from sentence_transformers import SentenceTransformer
# from BCEmbedding import RerankerModel


app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conan = SentenceTransformer("/dataset/CG_rag_model/git_repo/Conan")
@app.route("/embedding_conan", methods=["GET", "POST"])
def embedding_conan():
    #print(request.json)
    segments: List[str] = request.json["segments"]
    #segments_embeddings = conan.encode(segments, 
    #                        max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    #                        )['dense_vecs']
    segments_embeddings = conan.encode(segments)
    return segments_embeddings.tolist()

if __name__ == "__main__":
    # query = "今天星期几？"
    # refs = ["今天是教师节", "今天礼拜二", "你好"]
    # print(get_rerank_scores(query, refs))
    app.run(host="0.0.0.0", port=18848, debug=True, use_reloader=False)
