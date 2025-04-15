## FactScore
The repository is an enhanced version of the [original factscore](https://github.com/shmsw25/FActScore), an evaluation pipeline that breaks an LLM generation into a series of atomic facts and computes the percentage of the facts supported by the provided database.
See more about the pipeline in factscorer.py.

The changes made are significant:
1. matching titles from the database are searched by embedding distances rather than by character-level comparison as in the original.
2. you may not pass the topic of the generation explicitly: the titles will be found for each generation fact itself.
3. all API requests are sent asynchronously that dramatically reduces the execution time to, for example, 5.5 seconds for a batch of three one-sentence generations.
4. you can provide any knowledge source you want (one restriction: it should be a json-file), and the appropriate database will be created automatically

## Setup
**Prerequisites:**
*  json-file with the knowledge source information (see more on its format in database.py)
*  embeddings of the knowledge source titles and trained faiss index for them (see more in create_faiss_index.py)

Create a conda environment with python==3.12:
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

## Running factscore
1. create db-file with your json-file database:
2. evaluate generations with their topics known (then the titles of the articles will be found for the topics explicitly):
3. or unknown (then the titles of the articles will be found for the facts themselves):
4. on sentence-level (as in the original factscore):
5. or on paragraph-level:
