import argparse
import asyncio
import os
import sqlite3

import faiss
import numpy as np
import psutil
from faiss.contrib.ondisk import merge_ondisk
from loguru import logger
from tqdm import tqdm

from factscore.retrieval import APIEmbeddingFunction


class FAISSIndexCreator:
    def __init__(
        self,
        index_capacity,
        dim,
        embeds_dir,
        data_db,
        table_db,
        indexes_dir,
        model_name,
        total_len=None,
    ):
        """
        Builds an on-disk FAISS IVF index by creating and merging sharded indexes
        to reduce RAM usage while searching the embeddings. In particular:
        1. gets embeddings to the all db titles using API and saves them into self.num_indexes chunks,
        each chunk containing index_capacity embeddings (except of the last that contains the remaining embeds)
        2. trains an IVF index faiss_index_type on as many as possible embeddings chunks (that fit into RAM)
        3. adds each embedding chunk into the corresponding shard index
        4. merge the shard indexes to create the final index that will be used in FActScore-turbo pipeline
        Useful links:
        - https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM,
        - https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py
        """
        self.index_capacity = index_capacity
        self.dim = dim
        self.embeds_dir, self.indexes_dir = embeds_dir, indexes_dir
        if not os.path.exists(self.embeds_dir):
            os.mkdir(self.embeds_dir)
        if not os.path.exists(self.indexes_dir):
            os.mkdir(self.indexes_dir)
        self.data_db, self.table_db = data_db, table_db
        if total_len is not None:
            self.total_len = total_len
        else:
            connection = sqlite3.connect(data_db)
            cursor = connection.cursor()
            results = cursor.execute(f"SELECT count(title) FROM {table_db}")
            self.total_len = results.fetchall()[0][0]
            cursor.close()
        logger.info(f"Total titles: {self.total_len}")

        self.num_indexes = (
            self.total_len // index_capacity + 1
        )  # if self.total_len % index_capacity != 0 else self.total_len // index_capacity
        self.ef = APIEmbeddingFunction(model_name=model_name, dimensions=dim)

    async def _save_embeds_for_shard(
        self, start, shard, shard_is_final=False, batch_size=1_000
    ):
        """
        Computes embeddings to the titles with ids from <start> to <start + index_capacity> and
        saves them on the current embeddings chunk.
        Args:
            start: from what id to start adding vectors in the index
            shard: number of the current idx
            shard_is_final: if the current idx is final
        """
        connection = sqlite3.connect(self.data_db)
        cursor = connection.cursor()
        results = cursor.execute(f"SELECT title FROM {self.table_db}")
        titles = results.fetchall()
        vecs_chunk = []
        logger.info(f"Start getting embeds for shard {shard}")
        for i in tqdm(
            range(start, min(start + self.index_capacity, len(titles)), batch_size),
            desc=f"Getting embeds for {shard} shard",
        ):
            ids = [j for j in range(i, min(i + batch_size, len(titles)))]
            titles_to_add = list(map(lambda x: str(x[0]), titles[ids[0] : ids[-1] + 1]))
            vecs_batch = await self.ef(titles_to_add)
            # vecs_batch = self.ef(titles_to_add, len(titles_to_add))
            vecs_batch = np.array(vecs_batch).astype(np.float32)
            if not shard_is_final and len(vecs_batch) != batch_size:
                raise ValueError(
                    f"batch size is {batch_size}, but got {len(vecs_batch)} embeddings"
                )
            vecs_chunk.extend(vecs_batch)
        vecs_chunk = np.array(vecs_chunk).astype(np.float32)
        with open(f"{self.embeds_dir}/embeds_block{shard}.npy", "wb+") as f:
            np.save(f, vecs_chunk)
        cursor.close()

    async def save_embeds(self):
        """
        Computes embeddings to the titles and saves them into self.num_indexes shards,
        each shard containing self.index_capacity embeddings.
        """
        start = 0
        for shard in range(self.num_indexes - 1):
            await self._save_embeds_for_shard(start, shard, False)
            start += self.index_capacity
        await self._save_embeds_for_shard(start, self.num_indexes - 1, True)
        logger.info("Saved all embeddings")

    def train_index(self, faiss_index_type):
        """
        Trains faiss_index_type index on as many shards as possible (that can fit into RAM)
        and saves it into self.indexes_dir/trained.index
        Args:
            faiss_index_type: which FAISS index to train (preferably IVF)
        """
        index = faiss.index_factory(self.dim, faiss_index_type)
        index = faiss.index_cpu_to_all_gpus(index)
        try:
            embeds_train = np.load(f"{self.embeds_dir}/embeds_block0.npy")
        except MemoryError:
            logger.error(
                "Couldn't fit any shard index in RAM. Please reduce index capacity and try again"
            )
            raise MemoryError(
                "Couldn't fit any shard index in RAM. Please reduce index capacity and try again"
            )
        available_memory = psutil.virtual_memory().available * 0.65  # in bytes
        embeds_chunk_memory = (4 * self.dim + 8) * self.index_capacity
        available_chunks_number = int(available_memory // embeds_chunk_memory)
        logger.info(
            f"""Available RAM: {available_memory / 2**20:.5f} MB
                    RAM needed for one shard index: {embeds_chunk_memory / 2**20:.5f} MB
                    Index will be trained on {min(available_chunks_number, self.num_indexes)} shard indexes"""
        )
        for i in range(1, min(available_chunks_number, self.num_indexes)):
            cur_chunk = np.load(f"{self.embeds_dir}/embeds_block{i}.npy")
            embeds_train = np.concatenate(
                (embeds_train, cur_chunk), axis=0, dtype=np.float16
            )
        logger.info("Start training index")
        index.train(embeds_train)
        index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, f"{self.indexes_dir}/trained.index")
        logger.info("Index trained")

    def _add_vectors_to_shard(self, start, shard):
        """
        Add i-th embeddings chunk to i-th shard index.
        Args:
            start: from what id to start adding vectors in the index
            shard: number of the current shard
        """
        index = faiss.read_index(f"{self.indexes_dir}/trained.index")
        logger.info(f"Start adding vectors to shard {shard}")
        ids = [
            j for j in range(start, min(start + self.index_capacity, self.total_len))
        ]
        vecs = np.load(f"{self.embeds_dir}/embeds_block{shard}.npy")
        index.add_with_ids(vecs, np.array(ids))
        faiss.write_index(index, f"{self.indexes_dir}/block_{shard}.index")

    def add_vectors_to_shards(self):
        """
        Adds embeddings into the corresponing shard indexes
        """
        start = 0
        for shard in range(self.num_indexes):
            self._add_vectors_to_shard(start, shard)
            start += self.index_capacity
        logger.info("Added all vectors to shards")

    def merge_sharded_indexes(self, final_index_name="all_vecs.index"):
        index = faiss.read_index(f"{self.indexes_dir}/trained.index")
        block_fnames = [
            f"{self.indexes_dir}/block_{bno}.index"
            for bno in range(1, self.num_indexes)
        ]
        merge_ondisk(index, block_fnames, self.indexes_dir + "merged_index.ivfdata")
        faiss.write_index(index, f"{self.indexes_dir}/{final_index_name}")
        logger.info("Merged all shards")


def build_on_disk_index(args):
    """
    Uses FAISSIndexCreator to create an on-disk IVF index of the titles embeddings.
    See more in FAISSIndexCreator.__init__
    """
    index_creator = FAISSIndexCreator(
        args.index_capacity,
        args.dim,
        args.embeds_dir,
        args.data_db,
        args.table_name,
        args.indexes_dir,
        args.model_name,
    )
    asyncio.run(index_creator.save_embeds())
    index_creator.train_index(args.faiss_index_type)
    index_creator.add_vectors_to_shards()
    index_creator.merge_sharded_indexes(args.final_index_name)
    logger.info("Successfully built on-disk index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_db", help="path to db file with titles and texts", required=True
    )
    parser.add_argument(
        "--table_name",
        help="name of the corresponding table in data_db",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        help="embedding model name to use",
        default="text-embedding-3-small",
    )
    parser.add_argument("--dim", help="dimension of the embeddings", default=1536)
    parser.add_argument(
        "--embeds_dir",
        help="path to the dir where the titles embeddings will be stored",
        default="embeds",
    )
    parser.add_argument(
        "--indexes_dir",
        help="path to the dir where the FAISS indexes will be created and sharded",
        default="faiss_indexes",
    )
    parser.add_argument(
        "--index_capacity",
        help="how many vectors to store in one sharded index",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--faiss_index_type", help="which FAISS index to train", default="IVF32768,Flat"
    )
    parser.add_argument(
        "--final_index_name",
        help="name of the final index that will be used in FActScore-turbo pipeline",
        default="embeddings.index",
    )
    args = parser.parse_args()
    build_on_disk_index(args)
