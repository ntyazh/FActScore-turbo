import numpy as np
from tqdm import tqdm
import os
import sqlite3
import faiss
from factscore.api_requests_processor import process_api_requests_from_list

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"


class APIEmbeddingFunction():
    def __init__(
            self,
            base_url,
            model_name,
            dimensions=1536
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.dimensions = dimensions

    async def __call__(self, input: list):
        requests = list(map(lambda x: {"input": x,
                                       "model": self.model_name,
                                       "dimensions": self.dimensions},
                            input))
        embeds, failed_results = await process_api_requests_from_list(requests,
                                                                      self.base_url,
                                                                      api_key=os.environ["EMBEDDINGS_API_KEY"],
                                                                      proxy=os.environ["EMBEDDINGS_PROXY"] if os.environ["EMBEDDINGS_PROXY"] != "None" else None,
                                                                      )
        return embeds, failed_results


class EmbedRetrieval:
    @classmethod
    async def create(cls,
                     embedding_base_url,
                     embedding_model_name,
                     embed_dimension,
                     faiss_index,
                     data_db,
                     table_name,
                     ):
        self = cls()
        self.ef = APIEmbeddingFunction(base_url=embedding_base_url,
                                       model_name=embedding_model_name,
                                       dimensions=embed_dimension)
        self.index = faiss.read_index(faiss_index)
        self.table_name = table_name
        self.connection = sqlite3.connect(data_db)
        return self


    async def search(self, query, k):
        '''
        find k titles with the closest embedding distance to the query
        '''
        assert isinstance(query, list)
        embed, _ = await self.ef(query)
        embed = np.array(embed)
        if len(embed.shape) == 1:
            embed = np.array(embed).reshape(1, -1)
        distances, ids = self.index.search(np.array(embed), k)
        ids, distances = ids[0], distances[0]
        texts, titles = [], []
        cursor = self.connection.cursor()
        for id in ids:
            cursor.execute(
                f"SELECT text, title FROM {self.table_name} WHERE id =" + (str(id + 1)))
            id_results = cursor.fetchall()
            id_texts, id_titles = id_results[0][0], id_results[0][1]
            texts.append(id_texts)
            titles.append(id_titles)
        cursor.close()
        return texts, titles


    async def get_texts_for_title(self, topic: str, k):
        '''
        returns the k texts closest by vector distance to the topic 
        (the distance is calculated between the topic and titles from wiki, not texts)
        '''
        texts, titles = await self.search([topic], k)
        results_chunks = []
        for i, text in enumerate(texts):
            results_chunks.extend([{"title": titles[i], "text": para}
                                for para in text.split(SPECIAL_SEPARATOR)])
        return results_chunks
