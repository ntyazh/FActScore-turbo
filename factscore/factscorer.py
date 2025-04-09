import os
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
                 sentence_level=False,
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
            model_name=completions_model_name,
            sentence_level=sentence_level)
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
                        # topics: list,
                        generations: list,
                        atomic_facts=None,
                        topics=None,
                        k=5):
        '''
        сonsistently computes factscore for each generation 
        as unweighted mean of scores for the all atomic facts in the generation

        topics: topics of the atomic facts (if they present, else topics=None)
        generations: the generations we try to score
        atomic_facts: if we already have atomic facts for the generations somehow
        k: how many articles we want to add to the RAG context
        '''
        assert isinstance(generations, list), "generations and topics must be lists"
        if self.cost_estimate:
            total_words = 0
            for gen in generations:
                total_words += await self.af_generator.run(gen, cost_estimate=self.cost_estimate)
            return total_words

        if atomic_facts is not None:
            assert len(topics) == len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            atomic_facts, char_level_spans = [], []
            outputs = await self.af_generator.run(generations)
            for facts_triplets in outputs:
                # (sentence/passage, [atomic facts from the sentence], [spans of the facts])
                generation_atomic_facts, generation_char_level_spans = [], dict()
                for triplet in facts_triplets:
                    if len(triplet[1]) > 0:
                        generation_atomic_facts.extend(triplet[1])
                        generation_char_level_spans.update(dict(zip(triplet[1], triplet[2])))
                atomic_facts.append(generation_atomic_facts)
                char_level_spans.append(generation_char_level_spans)
            assert len(atomic_facts) == len(generations), f"atomic facts should have the same length as generations, got: generations {len(generations)}, atomic facts {len(atomic_facts)}"

        scores = []
        decisions, passages = await self._get_score(atomic_facts, char_level_spans, topics, k=k)
        for generation_decisions in decisions:
            score = 0
            if len(generation_decisions) == 0:
                scores.append(score)
                continue
            for d in generation_decisions:
                score += int(d["is_supported"])
            score /= len(generation_decisions)
            scores.append(score)
        out = {
            "decisions": decisions,
            "scores": scores,
            # "passages": passages, # for debug, to check that the retrieved from wiki passages really contain appropriate information
        }
        return out

    
    async def _get_score(self, list_of_atomic_facts, list_of_char_level_spans, topics: list = None, k=2):
        '''
        Processes a batch of generations with a single lm.generate() call.
        Preserves the {fact: span} dictionary structure for each generation.

        Args:
            list_of_atomic_facts: List[List[str]] - List of fact lists (one per generation)
            list_of_char_level_spans: List[Dict[str, tuple]] - List of {fact: span} dicts (one per generation)
            topics: Optional[List[str]] - Topics for each generation
            k: RAG context passages per fact

        Returns:
            Tuple of (list_of_decisions, list_of_passages) where:
            - list_of_decisions: List[List[dict]] - Results per generation
            - list_of_passages: List[dict] - RAG passages per generation
        '''
        assert len(list_of_atomic_facts) == len(list_of_char_level_spans), \
            "Input lists must have equal length"
        if topics is not None:
            assert len(topics) == len(list_of_atomic_facts), \
                "Topics must match generations count"

        all_facts = []
        fact_origin = []  # tracks which generation each fact came from
        for gen_idx, facts in enumerate(list_of_atomic_facts):
            all_facts.extend(facts)
            fact_origin.extend([gen_idx] * len(facts))

        prompts, passages_for_atoms = await self.get_rag_prompts_and_passages(
            all_facts,
            topic=None if topics is None else topics[0],  
            k=k
        )
        outputs = await self.lm.generate([p[1] for p in prompts])

        decisions_by_generation = [[] for _ in range(len(list_of_atomic_facts))]
        for i, (output, (fact, _)) in enumerate(zip(outputs, prompts)):
            gen_idx = fact_origin[i]
            char_span = list_of_char_level_spans[gen_idx].get(fact, None)
            generated_answer = output.lower()
            if "true" in generated_answer and "false" in generated_answer:
                is_supported = generated_answer.index("true") > generated_answer.index("false")
            elif "true" in generated_answer:
                is_supported = True
            else:
                is_supported = False

            decisions_by_generation[gen_idx].append({
                "atom": fact,
                "is_supported": is_supported,
                "span": char_span
            })
        return decisions_by_generation, passages_for_atoms

    async def get_rag_prompts_and_passages(self, atomic_facts, topic:str = None, k=2):
        prompts = []
        texts = await self.db.embed_retrieval.get_texts_for_title(topic, k) if topic is not None else \
            await self.db.embed_retrieval.get_texts_for_title(atomic_facts, k)
        passages_for_atoms = {}
        for atom in atomic_facts:
            atom = atom.strip()
            if topic is not None:
                passages = self.db.get_bm25_passages(topic, atom, texts, k=k)
                passages_for_atoms[atom] = passages
                definition = f"Task: answer the question about {topic} based on the given context. \n\n"
            else:
                passages = self.db.get_bm25_passages(topic, atom, texts[atom], k=k)
                passages_for_atoms[atom] = passages
                definition = "Task: answer the question based on the given context. \n\n"
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
        return prompts, passages_for_atoms
