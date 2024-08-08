import pickle
import jieba
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import TextNode


class BM25Retriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        session_id,
        similarity_top_k: int = 5,
    ) -> None:
        """Init params."""
        with open("./bm25/" + session_id + "_bm25", "rb") as fin:
            self.bm25 = pickle.load(fin)
        with open("./bm25/" + session_id + "_text_chunks", "rb") as fin:
            self.data = pickle.load(fin)
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve."""
        doc_scores = self.bm25.get_scores(jieba.cut_for_search(query_bundle.query_str))
        nodes_with_scores = [
            NodeWithScore(node=TextNode(text=self.data["text_chunks"][i], metadata=self.data["nth_file_data"][i]), score=doc_scores[i])
            for i in range(len(self.data["text_chunks"]))]
        return nodes_with_scores
