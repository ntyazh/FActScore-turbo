import os
import sqlite3
from typing import Union

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from factscore.api_requests_processor import process_api_requests_from_list

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class APIEmbeddingFunction:
    def __init__(self, model_name: str, dimensions: int = 1536):
        """
        Function for sending requests to receive embeddings

        model_name: name of the embedding model to use
        dimensions: dimension of the embeddings
        """
        self.model_name = model_name
        self.dimensions = dimensions

    async def __call__(self, input_texts: list):
        """
        Returns:
        embeds: list of embeddings for the input_texts
        failed_results: list of raised exceptions if the requests were not successful
        """
        requests = list(
            map(
                lambda x: {
                    "input": x,
                    "model": self.model_name,
                    "dimensions": self.dimensions,
                },
                input_texts,
            )
        )
        embeds = await process_api_requests_from_list(
            requests,
            os.environ["EMBEDDINGS_BASE_URL"],
            api_key=os.environ["EMBEDDINGS_API_KEY"],
            proxy=(
                os.environ["EMBEDDINGS_PROXY"]
                if os.environ["EMBEDDINGS_PROXY"] != "None"
                else None
            ),
        )
        return embeds


class Retrieval:
    def __init__(
        self,
        embedding_model_name: str,
        faiss_index: str,
        data_db: str,
        table_name: str,
        embed_dimension: int = 1536,
    ):
        """
        Retrieves chunks of the texts in the database for the RAG-pipeline.

        Args:
            embedding_model_name: model to use to get embeddings (for example, text-embedding-3-small)
            faiss_index: path to the final IVF index with the all title embeddings (you can get it with create_faiss_index.py)
            data_db: path to the db file with the database
            table_name: name of the table with titles and texts in the database
            embedding_dimension: dimension of the title embedding
        """
        self.ef = APIEmbeddingFunction(
            model_name=embedding_model_name, dimensions=embed_dimension
        )
        self.index = faiss.read_index(faiss_index)
        self.table_name = table_name
        self.connection = sqlite3.connect(data_db)

    async def search_titles(self, queries: list[str], k: int) -> dict[str : list[str]]:
        """
        Searches for k titles from the database with the closest embedding distance to the query.
        Args:
            queries: list of either topics of the generations or atomic facts from the generations (depending on the topics argument in factscore.get_score)
            k: number of needed titles

        Returns:
            texts: dict {query: texts from the database with the found titles for the query}
            titles: dict {query: the k found titles for the query}
        """
        assert isinstance(queries, list)
        embed = await self.ef(queries)
        if len(embed) == 0:
            return [], []
        embed = np.array(embed)
        if len(embed.shape) == 1:
            embed = np.array(embed).reshape(1, -1)
        texts, titles = dict(), dict()
        cursor = self.connection.cursor()
        distances_queries, ids_queries = self.index.search(np.array(embed), k)
        for query, ids in zip(queries, ids_queries):
            cur_query_texts = []
            cur_query_titles = []
            for id in ids:
                cursor.execute(
                    f"SELECT text, title FROM {self.table_name} WHERE id ="
                    + (str(id + 1))
                )
                id_results = cursor.fetchall()
                id_texts, id_titles = id_results[0][0], id_results[0][1]
                cur_query_texts.append(id_texts)
                cur_query_titles.append(id_titles)
            texts[query] = cur_query_texts
            titles[query] = cur_query_titles
        cursor.close()
        return texts, titles

    async def get_texts_for_queries(
        self, queries: Union[str, list[str]], k: int
    ) -> dict[str, list[dict]]:
        """
        Searches for the k texts closest by embedding distance to the each topic
        Note: the distance is calculated between the topic and titles from the database, not texts

        Args:
            queries: topic of the generation or atomic facts from it depending on topics argument in factscore.get_score
            k: number of texts required
        Returns:
            dict {query: list of dicts with keys (title, chunk of the text with this title)}
        """
        assert isinstance(queries, str) or isinstance(queries, list)
        if isinstance(queries, str):
            queries = [queries]
        texts, titles = await self.search_titles(queries, k)
        if len(titles) == 0:
            return []

        results_chunks = dict()
        for query in texts.keys():
            query_results_chunks = []
            query_texts, query_titles = texts[query], titles[query]
            for i, text in enumerate(query_texts):
                query_results_chunks.extend(
                    [
                        {"title": query_titles[i], "text": para}
                        for para in text.split(SPECIAL_SEPARATOR)
                    ]
                )
            results_chunks[query] = query_results_chunks
        return results_chunks

    def get_bm25_passages(
        self, topic: str, fact: str, chunks: list[dict], n: int
    ) -> list[dict]:
        """
        Searches for n chunks that are most similar to the topic and fact using bm25

        Args:
            topic: topic of the generation (if specified)
            fact: fact from the generation
            chunks: list of dicts with keys (title, text) from which the n most appropriate chunks are going to be selected
            n: number of the appropriate chunks

        Returns:
            list of the n appropriate chunks
        """
        query = topic + " " + fact.strip() if topic is not None else fact.strip()
        bm25 = BM25Okapi(
            [
                text["text"].replace("<s>", "").replace("</s>", "").split()
                for text in chunks
            ]
        )
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:n]
        return [chunks[i] for i in indices]
