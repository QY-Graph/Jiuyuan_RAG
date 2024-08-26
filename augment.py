import os
import requests
import json
import time
from typing import List
import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request
from FlagEmbedding import BGEM3FlagModel, FlagReranker, FlagModel
from llama_index.core.vector_stores import VectorStoreQuery,MetadataFilter,MetadataFilters
from llama_index.core.vector_stores.types import FilterCondition,FilterOperator
from utils import log2ui
import jiuyuan_db
from jiuyuan_db.jiuyuan_vector_store import JiuyuanVectorStore


# from BCEmbedding import RerankerModel

app = Flask(__name__)


@app.route("/augment/user/<string:userid>/dialogue/<string:dialogueid>",methods=["POST"])
def augment(userid,dialogueid):
    func_start_time = time.time()
    #Get query embedding
    EMBEDDING_URL="http://localhost:8846/embedding_bge_m3"
    EMBEDDING_HEADERS=HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
    request_data = request.get_json()
    query = ""
    if 'message' in request_data:
        query = request_data['message']
    else:
        return app.response_class(
            json.dumps({"error":"请求中没有提供message等信息"}),
            status=400,
            mimetype='application/json'
        )
    embedding_start = time.time()
    r = requests.post(EMBEDDING_URL, json.dumps({'segments': query}), headers=EMBEDDING_HEADERS)
    embedding_end = time.time()
    embedding_overhead = embedding_end-embedding_start
    log2ui(f"Embedding time:{embedding_overhead}")
    query_embedding = json.loads(r.content)
    #filter_ = [("userid",userid),("dialogueid",dialogueid)] 
    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
          MetadataFilter(
            key="userid", operator=FilterOperator.EQ, value=userid
            ),
          MetadataFilter(
            key="dialogueid", operator=FilterOperator.EQ, value=dialogueid
            )
        ]
    )
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=10,
        filters = filters,
        mode="default"
    )

    #Get relevant chunks of text
    #db = JiuyuanVectorStore.PersistentClient(path=os.path.join(APP_DATA_PATH, "./chroma_db"))
    """
    JiuyuanVecdb_client = JiuyuanVecdb.Client(
        host='your_host_ip',
        port='your_port',
        user='your_username',
        password='your_password'
    )
    db = JiuyuanVecdb_client.database('your_database_name')
    """
    #vector_store = JiuyuanVectorStore(schema_name="public",table_name="vector_store_test_new", embed_dim=1024,\
    #    host='172.22.162.213', port='7474', user='default_user', password='', database_name='default_db')
    #vector_store = JiuyuanVectorStore(schema_name="public",table_name="test1035", embed_dim=1024,\
    #    host='172.22.162.213', port='7474', user='default_user', password='', database_name='default_db')
    vector_store = JiuyuanVectorStore(schema_name="public",table_name=dialogueid, embed_dim=1024,\
        host='172.22.162.13', port='7474', user='default_user', password='', database_name='default_db')
    #vector_store = JiuyuanVectorStore(chroma_collection=db.get_or_create_collection(f"{collection_id}"))
    #try:
    query_start = time.time()
    query_result = vector_store.query(vector_store_query)
    query_end = time.time()
    query_overhead = query_end - query_start
    log2ui(f"Query overhead:{query_overhead}")
    vector_store.close()
    #log2ui(f"Query size: {len(query_result)}")
    #vector_store.close()
    """
    except jiuyuan_db.jiuyuan_exception.JiuyuanException:
      return app.response_class(
          json.dumps({"error":"对应collection不存在"}),
          status=500,
          mimetype='application/json'
      )
    """
    refs = [node.text for node in query_result.nodes]
    log2ui(f"Query size: {len(refs)}, query res: {refs}")
    if(len(refs)==0):
      return app.response_class(
          json.dumps({"message":query,"info":"relevant results not found, returning original query."}),
        status=200,
        mimetype='application/json'
      )
    #log2ui(f"R_name :{sum}")

    #Start reranking
    RERANK_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
    RERANK_URL = "http://localhost:8846/get_rerank_scores_m3"
    rerank_start = time.time()
    r = requests.post(RERANK_URL, json.dumps({'query': query, 'refs': refs}), headers=RERANK_HEADERS)
    rerank_end = time.time()
    rerank_overhead = rerank_end-rerank_start
    log2ui(f"Rerank time:{rerank_overhead}")
    response = json.loads(r.content)
    response_pairs = [(res, i) for i, res in enumerate(response) if res > 0]
    TOPK = 5
    sorted_paires = sorted(response_pairs,key = lambda x : x[0],reverse=True)[0:min(TOPK,len(refs))]
    
    #Use top-k directly, test for no rerank
    #sorted_paires = refs[0:5]

    #Make prompt
    if len(sorted_paires)>0:
      topk_refs = "\n".join(f"第{i+1}篇文章:{refs[pair[1]]}\n" for i,pair in enumerate(sorted_paires))
      #topk_refs = "\n".join(f"第{i+1}篇文章:{pair}\n" for i,pair in enumerate(sorted_paires))
      prompt = f"请根据以下文章回答问题\n，{topk_refs}，问题是{query}"
    else:
      prompt = query
    response = app.response_class(
        json.dumps({"message":prompt}),
        status=200,
        mimetype='application/json'
    )
    func_end_time = time.time()
    func_overhead = func_end_time-func_start_time
    log2ui(f"total time :{func_overhead}")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1235, debug=True, use_reloader=False)
