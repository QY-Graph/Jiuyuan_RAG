import os
from local_llm import llm_nostream, use_prompt
from local_rerank import get_rerank_scores
from utils import log2ui, APP_DATA_PATH


class BasicStrategy:
    @staticmethod
    def init_summary(session_id, nth_file, doc_text, llm_model_str):
        content = use_prompt(
            system_prompt_file=None,
            prompt_file="./prompts/init_summary",
            variable_dict={"doc_text": doc_text})
        log2ui(f"~ init_summary: content=[{content}]")
        content = "<用户>" + content.strip("\n") + "<AI>"
        summary = llm_nostream(content=content, model_version=llm_model_str).strip()
        log2ui(f"~ init_summary: summary=[{summary}]")
        with open(os.path.join(APP_DATA_PATH, f"summary/{session_id}_{nth_file}"), "w") as fout:
            fout.write(summary)

    @staticmethod
    def get_retrieval_info(retrieval, query_str, session_id, current_files_num, similarity_top_k, embedding_model_str):
        log2ui(f"~ get_retrieval_info: retrieval begin")
        info = retrieval(
            query_str=query_str,
            session_id=session_id,
            nth_file=0,
            similarity_top_k=similarity_top_k * 6,
            embedding_model_str=embedding_model_str)
        refs = [node.text for node in info]
        log2ui(f"~ get_retrieval_info: refs=[{refs}]")
        log2ui(f"~ get_retrieval_info: rerank begin")
        rerank_scores = get_rerank_scores(query=query_str, refs=refs)
        rerank_index = sorted(range(len(rerank_scores)), key=lambda k: rerank_scores[k], reverse=True)
        # print(info)
        res = []
        for i in range(min(similarity_top_k, len(rerank_index))):
            node = info[rerank_index[i]]
            node.score = rerank_scores[rerank_index[i]]
            res.append(node)
        # print(res)
        log2ui(f"~ get_retrieval_info: rerank res=[{res}]")
        return res

    @staticmethod
    def gen_answer(session_id, query_str, info, current_files_num
                   # , llm_model_str
                   ):
        multidoc_info = ""
        refs = [[] for _ in range(current_files_num)]
        for node in info:
            # print(node)
            refs[node.metadata["nth_file"] - 1].append(node.text)
        MAX_LENGTH = 4000
        for nth_file in range(1, current_files_num + 1):
            with open(os.path.join(APP_DATA_PATH, f"summary/{session_id}_{nth_file}"), "w+") as fin:
                nth_file_summary = fin.read()
            new_data = f"已知第{str(nth_file)}篇文章内容：【{nth_file_summary}】，"
            new_data += f"最相关的细节信息为【{refs[nth_file - 1]}】；\n\n"
            if len(new_data) + len(multidoc_info) > MAX_LENGTH:
                break
            multidoc_info += new_data
        log2ui(f"~ gen_answer: multidoc_info=[{multidoc_info}]")
        content = use_prompt(
                    system_prompt_file="./prompts/system_prompt",
                    prompt_file="./prompts/gen_answer_multidoc",
                    variable_dict={"query_str": query_str, "multidoc_info": multidoc_info})
        content = "<用户>" + content.strip("\n") + "<AI>"
        print(f"gen_answer: content=[{content}]")
        log2ui(f"~ gen_answer: content=[{content}]")
        return content
        


    @staticmethod
    def get_raw_doc(session_id, nth_file):
        with open(os.path.join(APP_DATA_PATH, f"raw_doc_text/{session_id}_{nth_file}"), "r") as fin:
            return fin.read()
