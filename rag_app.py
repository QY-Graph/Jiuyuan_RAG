import os
import subprocess
import json
from pathlib import Path

#To remove a non-empty directory
import shutil

import chromadb
from flask import Flask, Response, request, jsonify
app = Flask(__name__)


import requests
import pickle
import jieba
from rank_bm25 import BM25Okapi

from llama_index.core import SimpleDirectoryReader
# from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, QueryBundle

from local_embedding import LocalBgeEmbedding
from local_retriever import VectorDBRetriever
from local_bm25 import BM25Retriever
from strategy.basic.basic_strategy import BasicStrategy
from local_llm import llm_nostream, llm, use_prompt
from utils import log2ui, APP_DATA_PATH, LocalSentenceSplitter
from jiuyuan_db.jiuyuan_vector_store import JiuyuanVectorStore
from jiuyuan_db.client.client import JiuyuanClient
from jiuyuan_db.jiuyuan_exception import JiuyuanException
from llama_index.core.vector_stores import VectorStoreQuery,MetadataFilter,MetadataFilters
from llama_index.core.vector_stores.types import FilterCondition,FilterOperator

embed_model = {
    "bge_large": LocalBgeEmbedding() # bge m3
}
db = chromadb.PersistentClient(path=os.path.join(APP_DATA_PATH, "./chroma_db"))
FILE_TPYE_FOR_SimpleDirectoryReader = ["txt", "pdf", "docx", "pptx", "ppt", "md"]


def doc_to_docx(doc_file, docx_file):
    subprocess.run(['unoconv', '-f', 'docx', '-o', docx_file, doc_file], 
                   env={'PYTHONPATH': "/usr/bin/python"})

def convert_doc_to_docx(input_file):
    output_file = input_file.replace(".doc", ".docx")
    command = ["/home/ceshi1/libreoffice/opt/libreoffice5.1/program/soffice", "--headless", "--convert-to", "docx", input_file,"--outdir",APP_DATA_PATH+"/"]
                                                                                              
    try:
        subprocess.run(command, check=True)
        # print(f"doc2docx conversion successful: {output_file}")
        print("doc2docx conversion successful: {}".format(output_file))

    except subprocess.CalledProcessError as e:
        print("Error during doc2docx conversion: {}".format(e))


def file2text(file_tpye, file_path, text_splitter):
    if file_tpye in FILE_TPYE_FOR_SimpleDirectoryReader:
        reader = SimpleDirectoryReader(input_files=[file_path])
        text_chunks = text_splitter.split_text("\n".join([d.text for d in reader.load_data()]))
    else:
        print(f"file_tpye not supported: [{file_tpye}]")
        #log2ui(f"!!! text_chunks=[{text_chunks}]")
    return text_chunks

def file2nodes(file_tpye, file_path, nth_file, text_splitter, embedding_model_str,metadata=None):
    text_chunks = file2text(file_tpye, file_path, text_splitter)
    log2ui(f"text_chunks size:{len(text_chunks)}")
    #print(f"text chunks: {text_chunks}")
    try:
      os.remove(file_path)
      log2ui(f"Try delete file path:{file_path}")
      folder_path = os.path.abspath(file_path)
      log2ui(f"Try delete file abs path :{folder_path}")
      #don't remove the directory with the consideration of delete conflict
      #folder = os.path.dirname(folder_path)
      #log2ui(f"Try delete folder:{folder}")
      #shutil.rmtree(folder)
      #os.rmdir(folder)
    except OSError as e:
      #print("delete error!")
      log2ui(f"Delete error:file path:{file_path}")
      return [TextNode(text="error")]
    
    URL = "http://localhost:8846/embedding_bge_m3"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
    r = requests.post(URL, json.dumps({'segments': text_chunks}), headers=HEADERS)
    embeddings = json.loads(r.content)
    return [TextNode(
            text=text_chunk,
            embedding=embeddings[i],
            metadata=metadata)
            for i,text_chunk in enumerate(text_chunks)]


def init_vector_store(session_id, nth_file, nodes):
    # vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection(f"{session_id}"))
    vector_store = JiuyuanVectorStore(schema_name="public",table_name=session_id, embed_dim=1024,
                                      host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    vector_store.add(nodes)
    vector_store.close()
    """
    for i in range(0,len(nodes),100):
      #log2ui(f"{nodes[i].text}")
      vector_store.add(nodes[i:min(len(nodes),i+100)])
      #TODO add some code here to support "progress" api, progress is a float number
      #db.UpdateOrCreate(userid,sessionid,docid,progress)
    """


def _doc_index(
        session_id,
        file_path,
        nth_file,
        file_tpye,
        strategy,
        embedding_model_str,
        chunk_size,
        chunk_overlap,
        metadata = None
        ):
    log2ui("~ file2nodes:" + f"session_id:[{session_id}]; #file:[{nth_file}]")

    text_splitter=LocalSentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    if strategy == "basic":
        nodes = file2nodes(
            file_tpye=file_tpye,
            file_path=file_path,
            nth_file=nth_file,
            text_splitter=text_splitter,
            embedding_model_str=embedding_model_str,
            metadata=metadata)

        log2ui("~ init_vector_store:" + f"session_id:[{session_id}]; #file:[{nth_file}]")
        init_vector_store(session_id, nth_file, nodes)

        # log2ui("~ init_summary:" + f"session_id:[{session_id}]; #file:[{nth_file}]")
        # BasicStrategy.init_summary(
        #     session_id=session_id,
        #     nth_file=nth_file,
        #     doc_text="".join(n.text for n in nodes),
        #     llm_model_str="8b") # NOTE:summary默认用8b模型

    elif strategy == "bm25rag":
        with open("./bm25/" + session_id + "_text_chunks", "rb") as fin:
            data = pickle.load(fin)
        text_chunks = file2text(file_tpye, file_path, text_splitter)
        text_chunks = data["text_chunks"].extend(text_chunks)
        bm25 = BM25Okapi([list(jieba.cut_for_search(d)) for d in text_chunks])
        with open("./bm25/" + session_id + "_text_chunks", "wb") as fout:
            pickle.dump({"text_chunks": text_chunks, "nth_file_data": nth_file}, fout)
        with open("./bm25/" + session_id + "_bm25", "wb") as fout:
            pickle.dump(bm25, fout)
        BasicStrategy.init_summary(
            session_id=session_id,
            nth_file=nth_file,
            doc_text="".join(n.text for n in nodes),
            llm_model_str="8b") # NOTE:summary默认用8b模型


def retrieval(query_str, session_id, nth_file, similarity_top_k, embedding_model_str):
    
    retriever = VectorDBRetriever(
        # vector_store=ChromaVectorStore(
        #     chroma_collection=db.get_collection(f"{session_id}")),
        vector_store= JiuyuanVectorStore(schema_name="public",table_name=session_id, embed_dim=1024,
                                      host='localhost', port='12321', user='jhan', password='', database_name='postgres'),
            embed_model=embed_model[embedding_model_str],
            query_mode="default",
            similarity_top_k=similarity_top_k)


    if embedding_model_str == "bge_large":
        nodes = retriever._retrieve(QueryBundle("为这个句子生成表示以用于检索相关文章：" + query_str))
    # print(nodes)
    return nodes


def _query(
        session_id,
        query_str,
        similarity_top_k,
        strategy,
        current_files_num,
        # llm_model_str,
        embedding_model_str,
        ):
        if strategy == "basic":
            info =  BasicStrategy.get_retrieval_info(
                retrieval, query_str, session_id, current_files_num, similarity_top_k, embedding_model_str)
            content = BasicStrategy.gen_answer(session_id, query_str, info, current_files_num
                                               # , llm_model_str
                                               )
            return content

@app.route("/index/progress/user/<string:userid>/dialogue/<string:dialogueid>/doc/<string:docid>",methods=["GET"])
def progress(userid,dialogueid,docid):
    #TODO
    #db.connetc(***)
    #progress = db.query(userid,dialogueid,docid)
    myclient = JiuyuanClient(host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    session = myclient.get_session()
    #query = "select progress from user_progress WHERE userid = "+userid+" AND sessionid = "+dialogueid+" AND docid ="+docid+";"
    query = f"select progress from user_progress WHERE userid = '{userid}' AND sessionid = '{dialogueid}' AND docid = '{docid}';"
    res = session.execute_sql(query)
    myclient.release_session(session)
    if(res == None):
      return app.response_class(
        json.dumps({"No result":"无结果"}),
        status=400,
        mimetype = 'application/json'
      )
    else: 
      res.next()
      progress = res.get_object(1)
      log2ui(f"--progress :{progress}")
      return app.response_class(
        json.dumps({"progress":progress}),
        status=200,
        mimetype = 'application/json'
      )

@app.route("/index/user/<string:userid>/dialogue/<string:dialogueid>", methods=["DELETE"])
def delete_session(userid,dialogueid):
    vector_store = JiuyuanVectorStore(schema_name="public",table_name=dialogueid, embed_dim=1024,
                                      host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
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
    #vector_store._initialize()
    vector_store._connect()
    vector_store.delete(filters)
    vector_store.close()
    myclient = JiuyuanClient(host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    session = myclient.get_session()
    query = f"DELETE FROM user_progress WHERE sessionid = '{dialogueid}';"
    try:
      session.execute_sql_update(query)
    except JiuyuanException as e:
      return app.response_class(
        json.dumps({"Delete":"Sql Execution Failed!"}),
        status=400,
        mimetype = 'application/json'
      )
    myclient.release_session(session)
    return app.response_class(
      json.dumps({"Delete":"success"}),
      status=200,
      mimetype = 'application/json'
    )

@app.route("/index/user/<string:userid>/dialogue/<string:dialogueid>/doc/<string:docid>", methods=["DELETE"])
def delete_doc(userid,dialogueid,docid):
    vector_store = JiuyuanVectorStore(schema_name="public",table_name=dialogueid, embed_dim=1024,
                                      host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    file_id = docid.split(".")[0]
    filters = MetadataFilters(
        condition=FilterCondition.AND,
        filters=[
          MetadataFilter(
            key="userid", operator=FilterOperator.EQ, value=userid
            ),
          MetadataFilter(
            key="dialogueid", operator=FilterOperator.EQ, value=dialogueid
            ),
          MetadataFilter(
            key="docid", operator=FilterOperator.EQ, value=file_id
            )
        ]
    )
    #vector_store._initialize()
    vector_store._connect()
    vector_store.delete(filters)
    vector_store.close()

    myclient = JiuyuanClient(host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    session = myclient.get_session()
    query = f"DELETE FROM user_progress WHERE docid = '{docid}';"
    try:
      session.execute_sql_update(query)
    except JiuyuanException as e:
      return app.response_class(
        json.dumps({"Delete":"Sql Execution Failed!"}),
        status=400,
        mimetype = 'application/json'
      )
    myclient.release_session(session)
    return app.response_class(
      json.dumps({"Delete":"success"}),
      status=200,
      mimetype = 'application/json'
    )
  
@app.route("/index/user/<string:userid>/dialogue/<string:dialogueid>/doc/<string:docid>", methods=["POST"])
def index(userid,dialogueid,docid):
    log2ui()
    log2ui("\n\n~~~\n\n~ doc_index begin:")
    #session_id: str = request.form["session_id"]
    session_id = dialogueid

    file_type = docid.split(".")[-1]
    file_id = docid.split(".")[0]
    file_url = f"http://172.22.162.216:8777/aichat/attachment/download?name=ATTACHMENT_{file_id}.{file_type}"
    #file_path = os.path.join(APP_DATA_PATH, dialogueid)
    file_path = APP_DATA_PATH+"/"
    #if not os.path.exists(file_path):
    #  os.makedirs(file_path)
         
    try:
      response = requests.get(file_url)
      response.raise_for_status()  # 确保请求成功
    except requests.RequestException as e:
      log2ui(f"Error downloading file: {e}")
      return app.response_class(
        json.dumps({"error":"Fail to download file!"}),
        status=400,
        mimetype = 'application/json'
      )

    file_name =f"{file_id}.{file_type}"
    file_ = os.path.join(file_path,file_name)
    try:
      with open(file_,'wb') as f:
        f.write(response.content)
    except IOError as e:
      log2ui(f"Error saving  file: {e}")
      return app.response_class(
        json.dumps({"error":"Fail to save downloaded file!"}),
        status=401,
        mimetype = 'application/json'
      )
    if file_type == "doc":
        convert_doc_to_docx(file_)
        file_type = "docx"
        file_ = file_ + "x"

    #TODO
    #can be infferred from "setting" in request body

    metadata = {"userid":userid,"dialogueid":dialogueid,"docid":file_id}
    """  
    _doc_index(
        session_id=dialogueid,
        file_path=file_,
        nth_file=1,
        file_tpye=file_type, 
        strategy="basic",
        embedding_model_str="bge_large", 
        chunk_size=512, 
        chunk_overlap=64,
        metadata=metadata
    )
    """
    text_splitter=LocalSentenceSplitter(
        chunk_size=512,
        chunk_overlap=646464646464)
    if file_type in FILE_TPYE_FOR_SimpleDirectoryReader:
        reader = SimpleDirectoryReader(input_files=[file_])
        text_chunks = text_splitter.split_text("\n".join([d.text for d in reader.load_data()]))
        log2ui(f"!!! text_chunks=[{text_chunks}]")
    else:
        log2ui(f"file_tpye not supported: [{file_tpye}]")

    try:
      os.remove(file_)
      log2ui(f"Try delete file path:{file_}")
      folder_path = os.path.abspath(file_)
      log2ui(f"Try delete file abs path :{folder_path}")
      #don't remove the directory with the consideration of delete conflict
      #folder = os.path.dirname(folder_path)
      #log2ui(f"Try delete folder:{folder}")
      #shutil.rmtree(folder)
      #os.rmdir(folder)
    except OSError as e:
      #print("delete error!")
      log2ui(f"Delete error:file path:{file_}")
      #return [TextNode(text="error")]
    
    URL = "http://localhost:8846/embedding_bge_m3"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    vector_store = JiuyuanVectorStore(schema_name="public",table_name=session_id, embed_dim=1024,
                                      host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    myclient = JiuyuanClient(host='172.22.162.11', port='7474', user='default_user', password='',database_name='default_db')
    session = myclient.get_session()
    #check if already inserted
    pre_query = f"select progress from user_progress WHERE userid = '{userid}' AND sessionid = '{dialogueid}' AND docid = '{docid}';"
    pre_res = session.execute_sql(pre_query)
    if(pre_res != None):
      return app.response_class(
        json.dumps({"Failure":"File already indexed!"}),
        status=402,
        mimetype = 'application/json'
      )
    #begin to insert nodes
    for i in range(0,len(text_chunks),100):
      progress =100* min(i+100,len(text_chunks))/len(text_chunks)
      texts = text_chunks[i:min(i+100,len(text_chunks))]
      r = requests.post(URL, json.dumps({'segments': texts}), headers=HEADERS)
      embeddings = json.loads(r.content)
      nodes = [TextNode(
              #text=text_chunk[i,min(i+100,len(text_chunks)],
              text = text_chunk,
              embedding = embeddings[i],
              metadata = metadata)
              for i,text_chunk in enumerate(texts)]
      vector_store.add(nodes)
      #update progress
      query = f"INSERT INTO user_progress (userid, sessionid, docid, progress) VALUES ('{userid}', '{dialogueid}', '{docid}', {progress}) ON CONFLICT (userid, sessionid, docid) DO UPDATE SET progress = EXCLUDED.progress;"
      res = session.execute_sql_update(query)
      myclient.release_session(session)
    vector_store.close()
    #TODO ask if the dbclient has close() API
    """
    return [TextNode(
            text=text_chunk,
            embedding=embeddings[i],
            metadata=metadata)
            for i,text_chunk in enumerate(text_chunks)]
    """
    resp = jsonify(success=True)
    return resp
    # return {"status": "ok", "logs": log2ui()}


@app.route("/query", methods=["GET", "POST"])
def query():
    log2ui()
    log2ui("\n\n~~~\n\n~ query begin:")
    session_id: str = request.json["session_id"]
    query_str: str = request.json["query_str"]
    current_files_num: int = int(request.json["current_files_num"])
    settings: dict = request.json["settings"]
    def generate_events(session_id, query_str):
        content = _query(
                    session_id=session_id,
                    query_str=query_str,
                    current_files_num=current_files_num,
                    strategy=settings["rag_strategy"],
                    similarity_top_k=settings["rag_retrieval_top"],
                    # llm_model_str=settings["rag_llm_model"],
                    embedding_model_str=settings["rag_embedding_model"])
        log2ui(f"~ query: final content to llm=[{content}]")
        # return content
        for r in llm(content=content, model_version=settings["rag_llm_model"]):
            yield "data: " + r + "\n\n"
        yield "event: log\ndata: " + json.dumps({"text": log2ui()}, ensure_ascii=False) + "\n\n"
    return Response(generate_events(session_id, query_str), mimetype="text/event-stream")


@app.route("/get_session_name", methods=["GET", "POST"])
def get_session_name():
    query_str: str = request.json["query_str"]
    settings: dict = request.json["settings"]
    content = use_prompt(
            system_prompt_file=None,
            prompt_file="./prompts/get_session_name",
            variable_dict={"query_str": query_str})
    # print(content)
    name = llm_nostream(content=content, model_version=settings["rag_llm_model"]).strip()
    # print(name)
    return name


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8099, debug=True, use_reloader=False)
