## FactScore-turbo
The repository is an enhanced version of the [original factscore](https://github.com/shmsw25/FActScore), an evaluation pipeline that breaks an LLM generation into a series of atomic facts and computes the percentage of the facts supported by the provided database. 

``FactScore-turbo`` extends this framework with critical upgrades for LLM training workflows. Moreover, it has been already used to improve the factual accuracy of SmolLM2-360M-Instruct. 
See more about the pipeline details in [``factscorer.py``](https://github.com/ntyazh/factscore/blob/main/factscore/factscorer.py).

**Key improvements:**
1. the pipeline accelerates fact checking by ~6.5 times due to asynchronous API queries and batchization.
2. provides much more reliable, stable and fast search of documents from the database by adding vector sharded FAISS index: matching titles from the database are searched by embedding distances rather than by character-level comparison as in the original.
3. supports adding arbitrary database.
4. has a much more user-friendly user interface.
5. one may not pass the topic of the generation explicitly: the titles will be found for each generation fact itself.

## Setup
**Prerequisites:**
* json-file with the knowledge source information of the format:
```json
{"title": TITLE_1, "text": TEXT_1}
{"title": TITLE_2, "text": TEXT_2}
...
```
* embeddings of the knowledge source titles and trained FAISS index for them

1. Install the requirements:
```bash
cd factscore
pip install -r requirements.txt
```
2. Create/Use the knowledge source database

* From json-file using [``database.py``](https://github.com/ntyazh/factscore/blob/main/factscore/database.py) (creates new db-file):

```python
from factscore.database import DocDB

doc_db = DocDB('desired-path-to-db', 'desired-table-name-in-db', 'path-to-json', size_of_chunks)
```

* For existing DBs: ensure the table has three columns: id, title, text

3. As the pipeline uses Embedding and ChatCompletion API, setup the base urls, API keys and proxies in the corresponding environment variables:
```bash
export EMBEDDINGS_API_KEY="your-key-for-embeddings"
export COMPLETIONS_API_KEY="your-key-for-completions"

export EMBEDDINGS_BASE_URL="https://your-embeddings-api.url"
export COMPLETIONS_BASE_URL="https://your-completions-api.url"

export EMBEDDINGS_PROXY="your-embeddings-proxy" # if not needed, pass "None"
export COMPLETIONS_PROXY="your-completions-proxy" # if not needed, pass "None"
```


4. **[optional]** Create a sharded index from the FAISS index to reduce RAM usage with [create_faiss_index.py](https://github.com/ntyazh/factscore/blob/main/factscore/create_faiss_index.py)

## Running factscore


1. Create factscore instance and register its knowledge source:

```python
from factscore.factscorer import factscorer

fs = FactScorer(completions_model_name="gpt-4o-mini",
                embeddings_model_name="text-embedding-3-small")
fs.register_knowledge_source(data_db="path-to-db",
                            table_name="table-name-in-db",
                            faiss_index="path-to-faiss-index"
                            )

```

2. Evaluate generations with their topics known (then the titles of the articles will be found for the topics explicitly):

```python
results = fs.get_score(generations=[generation1, generation2, ...], 
                       topics=[topic1, topic2, ...], 
                       k="number-of-articles-to-find", 
                       n="number-of-chunks")
```

or unknown (then the titles of the articles will be found for the facts themselves):

```python
results = fs.get_score(generations=[generation1, generation2, ...], 
                       k="number-of-articles-to-find", 
                       n="number-of-chunks")
```


