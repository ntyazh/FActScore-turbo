import numpy as np
import os
import torch
from factscore.atomic_facts import AtomicFactGenerator
from factscore.completions_llm import CompletionsLLM
from factscore.database import DocDB
import os

class FactScorer(object):
    def __init__(self,
                 completions_request_url,
                 completions_model_name,
                 embedding_request_url="https://api.openai.com/v1/embeddings",
                 embedding_model_name="text-embedding-3-small",
                 cache_dir=".cache/factscore",
                 cost_estimate=False,
                 batch_size=4):
        '''
        batch_size: batch_size of requests to llm while scoring facts (we score facts asynchronously)
        cache_dir: in the original fatscore there was a caching mechanism: 
        all prompts and answers were put in json file so that we didn't have to send the same prompt many times,
        but now it needs some fixes 
        '''
        self.model_name = completions_model_name
        self.embedding_base_url, self.embedding_model_name = embedding_request_url, embedding_model_name
        self.batch_size = batch_size

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cost_estimate = cost_estimate
        self.af_generator = AtomicFactGenerator(
            request_url=completions_request_url,
            model_name=completions_model_name)
        self.lm = CompletionsLLM(
            completions_model_name=completions_model_name,
            completions_request_url=completions_request_url)


    async def register_knowledge_source(
            self,
            faiss_index, 
            data_db,
            table_name,
            embedding_dimension=1536,
            max_passage_length=256):
        '''
        creates DocDB and EmbedRetrieval instances

        faiss_index: path to the final IVF index with the all title embeddings (you can get it with create_faiss_index.py)
        data_db: file to .db file
        table_name: name of the appropriate table  with columns (id, title, text) in the data_db
        embedding_dimension: dim of the title embeddings
        max_passage_length: max size of data chunks that we create while building db (db is already built so it's unimportant now)
        '''
        self.db = await DocDB.create(embedding_base_url=self.embedding_base_url,
                                     embedding_model_name=self.embedding_model_name,
                                     embedding_dimension=embedding_dimension,
                                     faiss_index=faiss_index,
                                     max_passage_length=max_passage_length,
                                     data_db=data_db,
                                     table_name=table_name)
        print("DB registered")


    async def get_score(self,
                        topics: list,
                        generations: list,
                        atomic_facts=None,
                        k=5):
        '''
        сonsistently computes factscore for each generation 
        as unweighted mean of scores for the all atomic facts in the generation

        topics: topics of the generations
        generations: the generations we try to score
        atomic_facts: if we already have atomic facts for the generations somehow
        k: how many articles we want to add to the RAG context
        '''
        assert isinstance(topics, list) and isinstance(
            generations, list), "generations and topics must be lists"
        if self.cost_estimate:
            total_words = 0
            for gen in generations:
                total_words += await self.af_generator.run(gen, cost_estimate=self.cost_estimate)
            return total_words

        if atomic_facts is not None:
            assert len(topics) == len(
                atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            atomic_facts, char_level_spans = [], []
            for topic, gen in zip(topics, generations):
                facts_triplets = await self.af_generator.run(gen) # triplets of the type 
                # (sentence, [atomic facts from the sentence], [spans of the facts])
                generation_atomic_facts, generation_char_level_spans = [], []
                for triplet in facts_triplets:
                    generation_atomic_facts.extend(triplet[1])
                    generation_char_level_spans.extend(triplet[2])
                atomic_facts.append(generation_atomic_facts)
                char_level_spans.append(generation_char_level_spans)

            assert len(atomic_facts) == len(topics)
            self.af_generator.save_cache()

        scores, decisions, passages = [], [], []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
                continue
            generation_decisions, generation_passages = await self._get_score(topic, facts, k=k)
            if len(generation_decisions) > 0:
                score = np.mean([d["is_supported"] for d in generation_decisions])
                decisions.append(generation_decisions)
                passages.append(generation_passages)
                scores.append(score)

        out = {
            "score": np.mean(scores) if len(scores) > 0 else 0.0,
            "decisions": decisions,
            "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None]),
            "scores": scores,
            "char_level_facts_spans": char_level_spans,
            "passages": passages, # for debug, to check that the retrieved from wiki passages really contain appropriate information
        }
        return out


    async def _get_score(self, topic, atomic_facts, k=2):
        '''
        gives score for one generation

        topic: topic of the generation
        atomic_facts: facts from the all sentences of the generation
        '''
        decisions = []
        prompts = []
        texts = await self.db.embed_retrieval.get_texts_for_title(topic, k)
        passages_for_atoms = {}
        for atom in atomic_facts:
            atom = atom.strip()
            passages = self.db.get_bm25_passages(topic, atom, texts, k=k)
            passages_for_atoms[atom] = passages
            definition = "Task: answer the question about {} based on the given context. \n\n".format(
                topic)
            context = ""
            for psg in reversed(passages):
                context += "Title: {}\nText: {}\n\n".format(
                    psg["title"], psg["text"].replace(
                        "<s>", "").replace(
                        "</s>", ""))
            definition += context.strip()
            prompt = "{}\n\nInput: {} True or False? Answer True if the information is supported by the context above and False otherwise.\nOutput:".format(
                definition.strip(), atom.strip())
            prompts.append((atom, prompt))

        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            curr_prompts = prompts[i:min(i + self.batch_size, len(prompts))]
            outputs.extend(await self.lm.generate([p[1] for p in curr_prompts]))

        for i, output in enumerate(outputs):
            atom = prompts[i][0]
            if isinstance(
                    output[1],
                    np.ndarray) or isinstance(
                    output[1],
                    torch.Tensor):
                # when logits are available
                logits = np.array(output[1])
                assert logits.shape[0] in [32000, 32001]
                true_score = logits[5852]
                false_score = logits[7700]
                is_supported = true_score > false_score
            else:
                # when logits are unavailable
                assert isinstance(
                    output, str), "output in _get_score must be string"
                generated_answer = output.lower()
                if "true" in generated_answer and "false" in generated_answer:
                    is_supported = generated_answer.index(
                        "true") > generated_answer.index("false")
                elif "true" in generated_answer:
                    is_supported = True
                elif "false" in generated_answer:
                    is_supported = False
                else:
                    is_supported = None
            decisions.append({"atom": atom, "is_supported": is_supported})
        return decisions, passages_for_atoms
