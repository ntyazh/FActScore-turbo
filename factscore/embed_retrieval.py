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


    async def search(self, queries, k):
        '''
        find k titles with the closest embedding distance to the query
        query: either topic of the generation or atomic facts from the generation
        '''
        assert isinstance(queries, list)
        embed, _ = await self.ef(queries)
        embed = np.array(embed)
        if len(embed.shape) == 1:
            embed = np.array(embed).reshape(1, -1)
        texts, titles = dict(), dict()
        cursor = self.connection.cursor()
        distances_queries, ids_queries = self.index.search(np.array(embed), k)
        # print("DISTS", distances_queries)
        for query, ids in zip(queries, ids_queries):
            cur_query_texts = []
            cur_query_titles = []
            for id in ids:
                cursor.execute(
                    f"SELECT text, title FROM {self.table_name} WHERE id =" + (str(id + 1)))
                id_results = cursor.fetchall()
                id_texts, id_titles = id_results[0][0], id_results[0][1]
                cur_query_texts.append(id_texts)
                cur_query_titles.append(id_titles)
            texts[query] = cur_query_texts
            titles[query] = cur_query_titles
        cursor.close()
        return texts, titles


    async def get_texts_for_title(self, topic, k):
        '''
        returns the k texts closest by vector distance to the topic 
        (the distance is calculated between the topic and titles from wiki, not texts)
        '''
        is_generation_topic = False
        if isinstance(topic, str):
            topic = [topic]
            is_generation_topic = True
        texts, titles = await self.search(topic, k)

        results_chunks = dict()
        for query in texts.keys():
            query_results_chunks = []
            texts_, titles_ = texts[query], titles[query]
            for i, text in enumerate(texts_):
                query_results_chunks.extend([{"title": titles_[i], "text": para}
                                for para in text.split(SPECIAL_SEPARATOR)])
            results_chunks[query] = query_results_chunks
        if is_generation_topic:
            return query_results_chunks
        return results_chunks
