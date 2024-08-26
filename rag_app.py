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

embed_model = {
    "bge_large": LocalBgeEmbedding() # bge m3
}
db = chromadb.PersistentClient(path=os.path.join(APP_DATA_PATH, "./chroma_db"))
FILE_TPYE_FOR_SimpleDirectoryReader = ["txt", "pdf", "docx", "pptx", "ppt", "md"]


def doc_to_docx(doc_file, docx_file):
    subprocess.run(['unoconv', '-f', 'docx', '-o', docx_file, doc_file], 
                   env={'PYTHONPATH': "/usr/bin/python"})


def file2text(file_tpye, file_path, text_splitter):
    if file_tpye in FILE_TPYE_FOR_SimpleDirectoryReader:
        reader = SimpleDirectoryReader(input_files=[file_path])
        text_chunks = text_splitter.split_text("\n".join([d.text for d in reader.load_data()]))
    else:
        print(f"file_tpye not supported: [{file_tpye}]")
    log2ui(f"!!! text_chunks=[{text_chunks}]")
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
                                      host='172.22.162.13', port='7474', user='default_user', password='',database_name='default_db')
    """
    for i in range(0,len(nodes),1):
      log2ui(f"{nodes[i].text}")
      vector_store.add(nodes[i:min(len(nodes),i+1)])
    """
    vector_store.add(nodes)
    vector_store.close()


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


@app.route("/index/user/<string:userid>/dialogue/<string:dialogueid>/doc/<string:docid>", methods=["POST"])
def index(userid,dialogueid,docid):
    log2ui()
    log2ui("\n\n~~~\n\n~ doc_index begin:")
    #session_id: str = request.form["session_id"]
    session_id = dialogueid

    file_url = f"http://172.22.162.216:8777/aichat/attachment/download?name=ATTACHMENT_{docid}.txt"
    file_path = os.path.join(APP_DATA_PATH, dialogueid)
    if not os.path.exists(file_path):
      os.makedirs(file_path)
         
    try:
      response = requests.get(file_url)
      response.raise_for_status()  # 确保请求成功
    except requests.RequestException as e:
      log2ui(f"Error downloading file: {e}")
      return app.reponse_class(
        json.dumps({"error":"Fail to download file!"}),
        status=400,
        mimetype = 'application/json'
      )
    file_name =f"{docid}.txt"
    file_ = os.path.join(file_path,file_name)
    try:
      with open(file_,'wb') as f:
        f.write(response.content)
    except IOError as e:
      log2ui(f"Error saving  file: {e}")
      return app.reponse_class(
        json.dumps({"error":"Fail to save downloaded file!"}),
        status=401,
        mimetype = 'application/json'
      )
    #TODO
    #can be infferred from "setting" in request body

    metadata = {"userid":userid,"dialogueid":dialogueid,"docid":docid}
    file_type = "txt"
    if file_type == "txt":
      """
      #TODO
      #extend parameters by adding "settings" to request body
      _doc_index(
          session_id=session_id,
          file_path=file_path,
          nth_file=nth_file,
          file_tpye=file_tpye, 
          strategy=settings["rag_strategy"],
          embedding_model_str=settings["rag_embedding_model"], 
          chunk_size=settings["rag_chunk_size"], 
          chunk_overlap=settings["rag_chunk_overlap"]
      )
      """
      _doc_index(
          session_id=dialogueid,
          file_path=file_,
          nth_file=1,
          file_tpye="txt", 
          strategy="basic",
          embedding_model_str="bge_large", 
          chunk_size=512, 
          chunk_overlap=64,
          metadata=metadata
      )
    """
    # preceding_files_num: int = int(request.form["preceding_files_num"])
    preceding_files_num: int = 0
    settings: dict = json.loads(request.form["settings"])
    files = request.files.getlist('file')
    for local_nth_file, f in enumerate(files):
        nth_file = preceding_files_num + local_nth_file + 1
        file_tpye = f.filename.split(".")[-1]
        Path(os.path.join(APP_DATA_PATH, "received_files")).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(APP_DATA_PATH, "received_files", f"{session_id}_{str(nth_file)}.{file_tpye}")
        f.save(file_path)
        if file_tpye == "doc":
            doc_to_docx(doc_file=file_path, docx_file=file_path + "x")
            file_tpye = "docx"
            file_path = file_path + "x"
        log2ui("~ doc_index:" + f"session_id:[{session_id}]; #file:[{nth_file}]; filename:[{f.filename}]")
        _doc_index(
            session_id=session_id,
            file_path=file_path,
            nth_file=nth_file,
            file_tpye=file_tpye, 
            strategy=settings["rag_strategy"],
            embedding_model_str=settings["rag_embedding_model"], 
            chunk_size=settings["rag_chunk_size"], 
            chunk_overlap=settings["rag_chunk_overlap"]
        )
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
    app.run(host="0.0.0.0", port=8091, debug=True, use_reloader=False)
