import re
import logging
# logging.basicConfig(filename='log', level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    handlers=[logging.FileHandler('log', 'a', 'utf-8')]
)

# APP_DATA_PATH = "/llm_demo_trans/wangyixuan/llm_demo/rag_demo-master/app_data"

APP_DATA_PATH = "./app_data"

LOG2UI_CACHE_FILEPATH = "./LOG2UI_CACHE_FILE.txt"
with open(LOG2UI_CACHE_FILEPATH, "w"):
    pass
def log2ui(logstr=""):
    if logstr == "":
        with open(LOG2UI_CACHE_FILEPATH, "r", encoding='utf-8') as fin:
            logs = fin.read()
        with open(LOG2UI_CACHE_FILEPATH, "w", encoding='utf-8'):
            pass
        return logs
    else:
        logging.debug(logstr)
        with open(LOG2UI_CACHE_FILEPATH, "a", encoding='utf-8') as fout:
            fout.write(logstr + "\n")
        return "ok"

class LocalSentenceSplitter:
    def __init__(self, chunk_size, chunk_overlap) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _zng(paragraph):
        pattern = u'([！？。…!?.\n])'
        return [sent for sent in re.split(pattern, paragraph, flags=re.U)]

    @staticmethod
    def _zng_new(paragraph):
        pattern = u'([！？。…!?.\n])'
        # return [sent for sent in re.split(pattern, paragraph, flags=re.U)]
        sentences = []
        sent_now = ''
        # 将句子与标点合并
        for s in re.split(pattern, paragraph, flags=re.U):
            if sent_now == '':
                sent_now = s
            elif len(s) <= len("！？"):
                sent_now += s
            else:
                sentences.append(sent_now)
                sent_now = s
        if sent_now != '':
            sentences.append(sent_now)
        return sentences

    def split_text(self, segment):
        chunks, chunk_now, size_now = [], [], 0
        no_left = False
        for s in LocalSentenceSplitter._zng_new(segment):
            no_left = False
            chunk_now.append(s)
            size_now += len(s)
            if size_now > self.chunk_size:
                chunk = "".join(chunk_now)
                chunk_now, size_now = self._get_overlap(chunk_now)
                chunks.append(chunk)
                no_left = True

        if no_left == False:
            chunks.append("".join(chunk_now))
        return chunks

    def _get_overlap(self, chunk):
        rchunk = chunk[:]
        rchunk.reverse()
        size_now, overlap = 0, []
        for s in rchunk[:-1]:
            overlap.append(s)
            size_now += len(s)
            if size_now > self.chunk_overlap:
                break
        overlap.reverse()
        return overlap, size_now

    @staticmethod
    def test__zng_new():

        # 测试段落中只有一个句子
        paragraph1 = "这是一个句子。"
        print("output:",LocalSentenceSplitter._zng_new(paragraph1))
        assert LocalSentenceSplitter._zng_new(paragraph1) == ["这是一个句子。"]

        # 测试段落中有多个句子
        paragraph2 = "这是一个句子。这是另一个句子。"
        print("output:",LocalSentenceSplitter._zng_new(paragraph2))
        assert LocalSentenceSplitter._zng_new(paragraph2) == ["这是一个句子。", "这是另一个句子。"]

        # 测试段落中有多个标点符号
        paragraph3 = "这是一个句子！这是另一个句子？"
        print("output:",LocalSentenceSplitter._zng_new(paragraph3))
        assert LocalSentenceSplitter._zng_new(paragraph3) == ["这是一个句子！", "这是另一个句子？"]

        # 测试段落中有换行符
        paragraph4 = "这是一个句子。\n这是另一个句子。"
        print("output:",LocalSentenceSplitter._zng_new(paragraph4))
        assert LocalSentenceSplitter._zng_new(paragraph4) == ["这是一个句子。\n", "这是另一个句子。"]


        print("All test cases pass")

if __name__ == "__main__":
    print(len("！？"))
    LocalSentenceSplitter.test__zng_new()
