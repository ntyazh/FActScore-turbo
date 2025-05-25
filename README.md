## FActScore-turbo
The repository is an enhanced version of the [original factscore](https://github.com/shmsw25/FActScore), an evaluation pipeline that breaks an LLM generation into a series of atomic facts and computes the percentage of the facts supported by the provided database. 

``FActScore-turbo`` extends this framework with critical upgrades for LLM training workflows. Moreover, it has been already used to improve the factual accuracy of SmolLM2-360M-Instruct. 
See more about the pipeline details in [``factscorer.py``](https://github.com/ntyazh/factscore/blob/main/factscore/factscorer.py).

**Key improvements:**
1. the pipeline accelerates fact checking by ~6.5 times (with batch size=1) due to asynchronous API queries and batchization. for further acceleration, increase the batch size: for example, with batch_size=128, the pipeline achieves 95.7× speedup.
2. provides much more reliable, stable and fast search of documents from the database by adding vector sharded FAISS index: matching titles from the database are searched by embedding distances rather than by character-level comparison as in the original.
3. supports adding arbitrary database.
4. has a much more user-friendly user interface.
5. one may not pass the topic of the generation explicitly: the titles will be found for each generation fact itself.

## Setup
**Prerequisites:**
* json-file with the knowledge source information of the format:
```
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

* From json-file using [``scripts/create_database.py``](https://github.com/ntyazh/factscore/blob/main/scripts/create_database.py) (creates new db-file):

```bash
python3 scripts/create_database.py \
    --data_db="desired path to .db file" \
    --json_file="path to the json-file with data" \
    --max_passage_length="length of the each chunk"
```
Please see more in [``scripts/create_database.py``](https://github.com/ntyazh/factscore/blob/main/scripts/create_database.py)

* For existing DBs: ensure the table has three columns: id, title, text.
* If you don't have any DB, you can download a pre-built .db file containing the Wikipedia 2023 dump [here](https://disk.yandex.ru/d/vLpW5eGZ4bXfbQ).

3. Create/Use FAISS index with titles embeddings

* Create a sharded FAISS index to reduce RAM usage with [scripts/create_faiss_index.py](https://github.com/ntyazh/factscore/blob/main/scripts/create_faiss_index.py) (it will compute all the embeddings for titles and store them automatically)

```bash
export EMBEDDINGS_API_KEY="your-key-for-embeddings"
export EMBEDDINGS_PROXY="your-embeddings-proxy"
export EMBEDDINGS_BASE_URL="https://your-embeddings-api.url"
python3 scripts/shard_faiss_index.py \
    --data_db="path to .db file" \
    --faiss_index_type="IVF100,Flat" \
    --index_capacity=1000 \
```
Please see more in [scripts/create_faiss_index.py](https://github.com/ntyazh/factscore/blob/main/scripts/create_faiss_index.py)

* For existing FAISS index: make sure its IDs match the corresponding IDs from the database titles.
* You can download the FAISS index for pre-built Wikipedia 2023 dump from step 2 [here](https://disk.yandex.ru/d/snh1-bBLbifsqQ).

4. As the pipeline uses Embedding and ChatCompletion API, setup the base urls, API keys and proxies in the corresponding environment variables:
```bash
export EMBEDDINGS_API_KEY="your-key-for-embeddings"
export COMPLETIONS_API_KEY="your-key-for-completions"

export EMBEDDINGS_BASE_URL="https://your-embeddings-api.url"
export COMPLETIONS_BASE_URL="https://your-completions-api.url"

export EMBEDDINGS_PROXY="your-embeddings-proxy" # if not needed, pass "None"
export COMPLETIONS_PROXY="your-completions-proxy" # if not needed, pass "None"
```

## Running FActScore-turbo


1. Create factscore instance and register its knowledge source:

```python
from factscore.factscorer import factscorer

fs = FactScorer(completions_model_name="gpt-4o-mini",
                embeddings_model_name="text-embedding-3-small")
fs.register_knowledge_source(data_db="path-to-db",
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


