import argparse
import json
import sqlite3
import time

import numpy as np
from loguru import logger
from tqdm import tqdm
from transformers import RobertaTokenizer

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
TABLE_NAME = "documents"


def build_db(data_db: str, data_path: str, max_passage_length: int = 256):
    """
    Creates sqlite3 db data_db from json-file with data of the type {title: text}
    The db will have three columns: id, title, text, where text consists of chunks with the number of tokens max_passage_length,
    joined with SPECIAL_SEPARATOR

    Args:
        data_db: desired path to .db file with columns (id, title, text)
        data_path: path to the json-file
        max_passage_length: length of the each chunk (in tokens)
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    titles = set()
    output_lines = []
    lines_count, start_time = 0, time.time()
    connection = sqlite3.connect(data_db, check_same_thread=False)
    c = connection.cursor()
    c.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (id, title PRIMARY KEY, text)")

    with open(data_path, "r") as f:
        total_len = sum(1 for line in f)
        f.seek(0)
        for line in tqdm(f, desc="Building DB", total=total_len):
            if line == "\n":
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
                    passages.append(tokens[offset : offset + max_passage_length])
                    offset += max_passage_length

            psgs = [
                tokenizer.decode(tokens)
                for tokens in passages
                if np.sum([t not in [0, 2] for t in tokens]) > 0
            ]
            text = SPECIAL_SEPARATOR.join(psgs)
            output_lines.append((lines_count, title, text))
            lines_count += 1

            if len(output_lines) == 1000000:
                c.executemany(f"INSERT INTO {TABLE_NAME} VALUES (?,?,?)", output_lines)
                output_lines = []
                logger.info(
                    "Finish saving %dM documents (%dmin)"
                    % (lines_count / 1000000, (time.time() - start_time) / 60)
                )

    if len(output_lines) > 0:
        c.executemany(f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?)", output_lines)
        logger.info(
            "Finish saving %dM documents (%dmin)"
            % (lines_count / 1000000, (time.time() - start_time) / 60)
        )

    connection.commit()
    logger.info("Inserted titles:")
    logger.info(get_all_titles(connection, TABLE_NAME))
    connection.close()


def get_all_titles(connection):
    """
    Returns all titles from the db to verify insertion
    """
    cursor = connection.cursor()
    cursor.execute(f"SELECT title FROM {TABLE_NAME}")
    titles = cursor.fetchall()
    titles = "\n".join(list(map(lambda x: x[0], titles)))
    cursor.close()
    return titles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_db",
        help="desired path to .db file with columns (id, title, text)",
        required=True,
    )
    parser.add_argument(
        "--json_path",
        help="path to the json-file with data of the type {title: text}",
        required=True,
    )
    parser.add_argument(
        "--max_passage_length",
        help="length of the each chunk (in RobertaTokenizer tokens)",
        default=256,
    )
    args = parser.parse_args()
    build_db(args.data_db, args.json_path, args.max_passage_length)
