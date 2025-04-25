import os
import sqlite3

import faiss
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
from loguru import logger
from tqdm import tqdm

from factscore.retrieval import APIEmbeddingFunction

'''
Faiss supports storing IVF indexes in a file on disk and accessing the file on-the-fly.
The simplest approach to do that -- on-disk index.
The on-disk index is built by merging the sharded indexes into one big index.
Useful links: 
https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM,
https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py


data_db: path to db file
base_url: url for embeddings, default -- openai server
table_db: name of the table in the database db
index_capacity: how many vectors to store in one sharded index 
indexes_dir: path to dir where IVF indexes will be created and merged
trained_index_name: name of file with trained IVF index in indexes_dir
'''

os.environ['EMBEDDINGS_API_KEY'] = "your api key"
os.environ["EMBEDDINGS_PROXY"] = "your proxy"

indexes_dir = 'path/to/folder/with/indexes/'
base_url = "https://api.openai.com/v1/embeddings"
data_db = "/path/to/db"
table_db = "documents"
index_capacity = 500_000
trained_index_name = 'faiss.index'


async def get_embeddings(start, part, part_is_final=False):
    '''
    Computes embeddings to the titles with ids from <start> to <start + index_capacity> and loads them on the current index
    Before using this function, you should already have trained IVF index from faiss, for example:
    index = faiss.index_factory(1536, "IVF32768,Flat") (IVF index with 32768 Voronoi cells and no quantization)

    Args:
        start: from what id to start adding vectors in the index 
        part: number of the current idx
        part_is_final: if the current idx is final
    '''
    connection = sqlite3.connect(data_db)
    cursor = connection.cursor()
    results = cursor.execute(f"SELECT title FROM {table_db}")
    titles = results.fetchall()
    batch_size = 1_000
    ef = APIEmbeddingFunction(base_url=base_url,
                              model_name="text-embedding-3-small",
                              dimensions=1536)
    index = faiss.read_index(indexes_dir + trained_index_name)
    logger.info("Start adding vectors to the index")
    for i in tqdm(range(start, min(start + index_capacity, len(titles)), batch_size)):
        ids = [j for j in range(i, min(i + batch_size, len(titles)))]
        titles_to_add = list(map(lambda x: str(x[0]), titles[ids[0]: ids[-1] + 1]))
        vecs, _ = await ef(titles_to_add)
        if not part_is_final:
            assert len(vecs) == batch_size, f"batch size is {batch_size}, but got {len(vecs)} embeddings"
        vecs = np.array(vecs).astype(np.float16)
        index.add_with_ids(vecs, np.array(ids))
    faiss.write_index(index, indexes_dir + "block_%d.index" % part)


def merge_sharded_indexes(number_of_indexes, final_index_name="all_vecs.index"):
    '''
    Args:
        number_of_indexes: how many sharded indexes you have
        final_index_name: to what file the merged result will be saved
    '''
    index = faiss.read_index(indexes_dir + trained_index_name)
    block_fnames = [
        indexes_dir + "block_%d.index" % bno
        for bno in range(1, number_of_indexes)
    ]
    merge_ondisk(index, block_fnames, indexes_dir + "merged_index.ivfdata")
    logger.info("Writing " + indexes_dir + final_index_name)
    faiss.write_index(index, indexes_dir + final_index_name)
    