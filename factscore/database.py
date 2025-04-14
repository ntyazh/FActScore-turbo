import json
import time
import sqlite3
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
import os.path

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class DocDB:
    def __init__(self,
                 data_db: str,
                 table_name: str,
                 data_json: str = None,
                 max_passage_length: int = 256
                 ):
        '''
        Creates sqlite3 data db from the json-file of the type {title: text}, if it doesn't exist yet,
        otherwise just connects to it.
        The db has three columns: id, title, text, where text consists of chunks with the number of tokens max_passage_length, joined with SPECIAL_SEPARATOR

        Args:
            data_db: path to .db file with columns (id, title, text) 
            table_name: name of the corresponding table in the sqlite3 db
            data_json: path to json-file with data of the format {title: text} (if the db doesn't exist yet)
            max_passage_length: length of the each chunk (in tokens)
        '''
        self.connection = sqlite3.connect(data_db, check_same_thread=False)
        if not os.path.exists(data_db):
            self.build_db(data_json, data_db, max_passage_length)
        self.data_db = data_db
        self.table_name = table_name

    def build_db(self, data_path, max_passage_length):
        '''
        Creates the table with name self.db file specified in data_db from json-file with data of the type {title: text}
        The db has three columns: id, title, text, where text consists of chunks with the number of tokens max_passage_length,
        joined with SPECIAL_SEPARATOR

        Args:
            data_path: path to the json-file
            max_passage_length: length of the each chunk (in tokens)
        '''
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        lines_count, start_time = 0, time.time() # for logging
        c = self.connection.cursor()
        sql_creation = f"CREATE TABLE {self.table_name} (title PRIMARY KEY, text);"
        c.execute(sql_creation)

        with open(data_path, "r") as f:
            total_len = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(f, desc="Building DB", total=total_len):
                if line == '\n':
                    continue
                dp = json.loads(line)
                title, text = dp["title"], dp["text"]
                if title in titles or len(text.strip()) == 0:
                    continue
                titles.add(title)
                passages = [[]]
                tokens = tokenizer(text)["input_ids"]
                max_length = max_passage_length - len(passages[-1])
                if len(tokens) <= max_length:
                    passages[-1].extend(tokens)
                else:
                    passages[-1].extend(tokens[:max_length])
                    offset = max_length
                    while offset < len(tokens):
                        passages.append(tokens[offset:offset + max_passage_length])
                        offset += max_passage_length

                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((lines_count, title, text))
                lines_count += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (lines_count / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (lines_count / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.check_titles_inserted()
        self.connection.close()

    def check_titles_inserted(self):
        '''
        Prints all titles from the db to verify insertion
        '''
        cursor = self.connection.cursor()
        sql_selection = f"SELECT title FROM {self.table_name}"
        cursor.execute(sql_selection)
        titles = cursor.fetchall()  
        for title in titles:
            print(title) 
        cursor.close()
