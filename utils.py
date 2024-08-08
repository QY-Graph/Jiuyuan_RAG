import re
import logging
logging.basicConfig(filename='log', level=logging.DEBUG)


APP_DATA_PATH = "./app_data"


LOG2UI_CACHE_FILEPATH = "./LOG2UI_CACHE_FILE.txt"
with open(LOG2UI_CACHE_FILEPATH, "w"):
    pass
def log2ui(logstr=""):
    if logstr == "":
        with open(LOG2UI_CACHE_FILEPATH, "r") as fin:
            logs = fin.read()
        with open(LOG2UI_CACHE_FILEPATH, "w"):
            pass
        return logs
    else:
        logging.debug(logstr)
        with open(LOG2UI_CACHE_FILEPATH, "a") as fout:
            fout.write(logstr + "\n")
        return "ok"

class LocalSentenceSplitter:
    def __init__(self, chunk_size, chunk_overlap) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _zng(paragraph):
        return [sent for sent in re.split(u'(！|？|。|\\n)', paragraph, flags=re.U)]


    def split_text(self, segment):
        chunks, chunk_now, size_now = [], [], 0
        no_left = False
        for s in LocalSentenceSplitter._zng(segment):
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