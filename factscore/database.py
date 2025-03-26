import json
import time
import sqlite3
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
from factscore.embed_retrieval import EmbedRetrieval
from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class DocDB(object):
    @classmethod
    async def create(cls,
                     embedding_base_url,
                     embedding_model_name,
                     embedding_dimension,
                     faiss_index,
                     data_db,
                     table_name,
                     max_passage_length=256,
                     ):
        '''
        connects to data db and creates embeddings vector storage

        embedding_base_url: where to post embeddings requests
        embedding_dimension: default 1536
        embedding_model_name: default text-embedding-3-small
        faiss_index: trained IVF faiss index with the all embeddings
        data_db: path to .db file with columns (id, title, text)
        table_name: name of the appropriate table in the db
        max_passage_length: length of chunks to split the wiki text to (during .db creation)
        '''
        self = cls()
        self.data_db = data_db
        self.max_passage_length = max_passage_length
        self.embed_retrieval = await EmbedRetrieval.create(data_db=data_db,
                                                           table_name=table_name,
                                                           embedding_base_url=embedding_base_url,
                                                           embedding_model_name=embedding_model_name,
                                                           faiss_index=faiss_index,
                                                           embed_dimension=embedding_dimension)

        self.connection = sqlite3.connect(
            self.data_db, check_same_thread=False)
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return self


    def build_db(self, data_path, total_len=None):
        '''
        creates the .db file from json file with data of the type {title: article}
        '''
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        try:
            c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
        except sqlite3.OperationalError:
            pass
        with open(data_path, "r") as f:
            if total_len is None:
                total_len = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(f, desc="Building DB", total=total_len):
                if line == '\n':
                    continue
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                passages = [[]]
                if len(text.strip()) == 0:
                    continue
                tokens = tokenizer(text)["input_ids"]
                max_length = self.max_passage_length - len(passages[-1])
                if len(tokens) <= max_length:
                    passages[-1].extend(tokens)
                else:
                    passages[-1].extend(tokens[:max_length])
                    offset = max_length
                    while offset < len(tokens):
                        passages.append(
                            tokens[offset:offset + self.max_passage_length])
                        offset += self.max_passage_length

                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum(
                    [t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany(
                        "INSERT INTO documents VALUES (?,?)",
                        output_lines)
                    output_lines = []
                    print(
                        "Finish saving %dM documents (%dmin)" %
                        (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" %
                  (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.check_titles_inserted()
        self.connection.close()


    def check_titles_inserted(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT title FROM documents")
        titles = cursor.fetchall()  # Fetches all the titles from the 'documents' table
        for title in titles:
            print(title)  # Print each title to verify
        cursor.close()
        

    def get_bm25_passages(self, topic, question, texts, k):
        '''
        returns k passages (parts of the texts) most similar to the topic using bm25
        '''
        query = topic + " " + question.strip() if topic is not None else question.strip()
        bm25 = BM25Okapi([text["text"].replace("<s>", "").replace(
            "</s>", "").split() for text in texts])
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [texts[i] for i in indices]
