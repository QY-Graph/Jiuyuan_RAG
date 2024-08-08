import os
from typing import List
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request
from FlagEmbedding import BGEM3FlagModel, FlagReranker, FlagModel
# from BCEmbedding import RerankerModel


app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # bge_large
# EMBEDDING_MODEL_PATH = "/data2/wangyixuan/models/bge-large-zh-v1.5"
# # embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL_PATH)
# # @app.route("/embedding", methods=["get", "post"])
# # def embedding_local():
# #     segments: List[str] = request.json["segments"]
# #     segments_embeddings = embedding_model.encode(segments)
# #     return segments_embeddings.tolist()
# embedding_model = model = FlagModel(EMBEDDING_MODEL_PATH, 
#                   query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
#                   use_fp16=True)
# @app.route("/embedding", methods=["get", "post"])
# def embedding_local():
#     segments: List[str] = request.json["segments"]
#     segments_embeddings = embedding_model.encode(segments)
#     return segments_embeddings.tolist()

# bge_m3
bge_m3 = BGEM3FlagModel('/home/jhan/github/bge-m3',
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
@app.route("/embedding_bge_m3", methods=["get", "post"])
def embedding_bge_m3():
    print(request.json)
    segments: List[str] = request.json["segments"]
    segments_embeddings = bge_m3.encode(segments, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    return segments_embeddings.tolist()


# # bge rerank large
# model_path = "/data2/wangyixuan/models/bge-reranker-large"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# model = model.to('cuda')
# model.eval()
# @app.route("/get_rerank_scores", methods=["get", "post"])
# def get_rerank_scores():
#     query: str = request.json["query"]
#     refs: List[str] = request.json["refs"]
#     with torch.no_grad():
#         pairs = [[query, ref] for ref in refs]
#         inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
#         scores = model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
#         return scores


# bge rerank m3
reranker_m3 = FlagReranker('/home/jhan/github/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
@app.route("/get_rerank_scores_m3", methods=["get", "post"])
def get_rerank_scores_m3(): 
    query: str = request.json["query"]
    refs: List[str] = request.json["refs"]
    return reranker_m3.compute_score([[query, ref] for ref in refs])


# # bce rerank
# bce_rerank_model = RerankerModel(model_name_or_path="/data2/wangyixuan/models/bce-reranker-base_v1")
# @app.route("/get_citation_score", methods=["get", "post"])
# def get_citation_score(): 
#     query: str = request.json["query"]
#     refs: List[str] = request.json["refs"]
#     sentence_pairs = [[query, ref] for ref in refs]
#     scores = bce_rerank_model.compute_score(sentence_pairs)
#     return scores



if __name__ == "__main__":
    # query = "今天星期几？"
    # refs = ["今天是教师节", "今天礼拜二", "你好"]
    # print(get_rerank_scores(query, refs))
    app.run(host="localhost", port=8846, debug=True, use_reloader=False)