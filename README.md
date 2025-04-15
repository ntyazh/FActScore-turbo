## FactScore
The repository is an enhanced version of the [original factscore](https://github.com/shmsw25/FActScore), an evaluation pipeline that breaks an LLM generation into a series of atomic facts and computes the percentage of the facts supported by the provided database.
See more about the pipeline in factscorer.py.

The changes made are significant:
1. matching titles from the database are searched by embedding distances rather than by character-level comparison as in the original.
2. you may not pass the topic of the generation explicitly: the titles will be found for each generation fact itself.
3. all API requests are sent asynchronously that dramatically reduces the execution time to, for example, 5.5 seconds for a batch of three one-sentence generations.
4. you can provide any knowledge source you want, and the appropriate database will be created automatically

## Setup
**Prerequisites:**
*  json-file with the knowledge source information (see more on its format in database.py)
*  embeddings of the knowledge source titles and trained faiss index for them (see more in create_faiss_index.py)

Create a conda environment with python3.12:
```
conda create -n factscore python==3.12
```
Clone the repository:
```
git clone https://github.com/ntyazh/factscore.git
```
Install the requirements:
```
cd factscore
pip install -r requirements.txt
```
Setup your API keys and proxies (if needed):
```
os.environ['EMBEDDINGS_API_KEY'] = "your api key for embedding model"
os.environ['COMPLETIONS_API_KEY'] = "your api key for completions model"
os.environ["EMBEDDINGS_PROXY"] = "your proxy for embedding model" (if not needed, pass "None")
os.environ['COMPLETIONS_PROXY'] = "your proxy for completions model" (if not needed, pass "None")
```
## Running factscore
<details>
<summary> Create db-file with your json-file database:
<br>
```
from factscore.database import DocDB
doc_db = DocDB('desired-path-to-db', 'desired-table-name-in-db', 'path-to-json', size_of_chunks)
```
</details>

<details>
<summary> Create factscore instance and register its knowledge source:
<br>
```
from factscore.factscorer import factscorer
fs = FactScorer(completions_request_url="for example, https://api.deepinfra.com/v1/openai/chat/completions",
                completions_model_name="for example, gpt-4o-mini",
                embeddings_request_url="for example, https://api.openai.com/v1/embeddings",
                embeddings_model_name="for example, text-embedding-3-small")
fs.register_knowledge_source(data_db="path-to-db",
                            table_name="table-name-in-db",
                            faiss_index="path-to-faiss-index"
                            )

```
</details>

<details>
<summary> Evaluate generations with their topics known (then the titles of the articles will be found for the topics explicitly):
<br>
```
results = fs.get_score(generations=[generation1, generation2, ...], 
                       topics=[topic1, topic2, ...], 
                       k="number-of-articles-to-find", 
                       n="number-of-chunks")
```
</details>

<details>
<summary> or unknown (then the titles of the articles will be found for the facts themselves):
<br>
```
results = fs.get_score(generations=[generation1, generation2, ...], 
                       k="number-of-articles-to-find", 
                       n="number-of-chunks")
```
</details>


