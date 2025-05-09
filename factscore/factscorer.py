from loguru import logger

from factscore.atomic_facts import AtomicFactGenerator
from factscore.completions_llm import CompletionsLLM
from factscore.retrieval import Retrieval


class FactScorer:
    def __init__(
        self, completions_model_name, embeddings_model_name, sentence_level=False
    ):
        """
        Computes factual score for a generation using the following pipeline:
        1. if sentence_level is True, splits the original generation into sentences (called passages),
        as it is done in the original factscore. otherwise, the passages is the generation itself.
        2. extracts independent atomic facts from the each passage with completions_model
        3. searches for the k closest to query (topic of the generation (if specified), or the fact itself) titles from the database using the embedding distances
        4. finds the n closest chunks of texts with these titles using bm25
        5. asks the completion llm if the fact is supported by these chunks. if it is, than the fact gets score 1, otherwise -1
        6. computes factual score for the generation as an unweighted mean of scores for the all facts this generation consists of.
        All requests are processed asynchronously and batched that speeds up the pipeline.

        Args:
            completions_model_name: model to use for the requests in the previous point
            embeddings_model_name: model to use for computing the embeddings
            sentence_level: whether to split a generation into sentences before the factual breakdown
        """
        self.embeddings_model_name = embeddings_model_name
        self.af_generator = AtomicFactGenerator(
            model_name=completions_model_name, sentence_level=sentence_level
        )
        self.lm = CompletionsLLM(completions_model_name=completions_model_name)

    def register_knowledge_source(
        self,
        faiss_index: str,
        data_db: str,
        table_name: str,
        embedding_dimension: int = 1536,
    ):
        """
        Creates DocDB and Retrieval instances

        faiss_index: path to the  IVF index with the all title embeddings (you can shard it to reduce RAM usage with scripts/create_faiss_index.py)
        data_db: path to .db file with the database (it can be created with scripts/create_database.py)
        table_name: name of the corresponding table  with columns (id, title, text) in the data_db
        embedding_dimension: dimension of the title embeddings
        """
        self.retrieval = Retrieval(
            data_db=data_db,
            table_name=table_name,
            embedding_model_name=self.embeddings_model_name,
            faiss_index=faiss_index,
            embed_dimension=embedding_dimension,
        )

    async def get_score(
        self,
        generations: list,
        atomic_facts: list = None,
        topics: list = None,
        k: int = 3,
        n: int = 5,
    ):
        """
        Asynchronously computes the factual for each generation by batches

        Args:
            generations: the generations to compute the factual scores for
            atomic_facts: atomic facts of the generations (if they are already extracted)
            topics: topics of the generations. if not None, the closest titles are searched for them, otherwise for the atomic facts themselves
            k: number of articles to find as the closest to the query (topic or atomic fact)
            n: number of chunks to include for the RAG-context for the each fact

        Returns:
            out: dict with keys decisions and scores, where
            - decisions: lists of dicts {atomic_fact, whether it is supported by the RAG-info, char_level span of the fact}
            for the each generation
            - scores: the factual scores for the generations
        """
        if not isinstance(generations, list):
            raise TypeError("generations must be a list")
        if atomic_facts is None:
            atomic_facts, char_level_spans = [], []
            outputs = await self.af_generator.run(generations)
            if len(outputs) == 0:
                return {"decisions": [], "scores": [0 for _ in range(len(generations))]}
            for facts_triplets in outputs:
                generation_atomic_facts, generation_char_level_spans = [], dict()
                for triplet in facts_triplets:
                    if len(triplet[1]) > 0:
                        generation_atomic_facts.extend(triplet[1])
                        generation_char_level_spans.update(
                            dict(zip(triplet[1], triplet[2]))
                        )
                atomic_facts.append(generation_atomic_facts)
                char_level_spans.append(generation_char_level_spans)
        if len(atomic_facts) != len(generations):
            raise ValueError(
                f"atomic facts should have the same length as generations, got: generations {len(generations)}, atomic facts {len(atomic_facts)}"
            )
        if topics is not None and len(topics) != len(atomic_facts):
            raise ValueError(
                f"atomic facts should have the same length as topics, got: topics {len(topics)}, atomic facts {len(atomic_facts)}"
            )
        scores = []
        decisions, passages = await self._get_score(
            atomic_facts, char_level_spans, k=k, n=n, topics=topics
        )
        if len(decisions) == 0:
            return {"decisions": [], "scores": [0 for _ in range(len(generations))]}
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
            "passages": passages,  # for debug, to check that the retrieved from the database passages really contain the appropriate information
        }
        return out

    async def _get_score(
        self,
        atomic_facts: list[list[str]],
        char_level_spans: list[dict],
        k: int,
        n: int,
        topics: list = None,
    ) -> tuple[list[list[dict]], list[dict]]:
        """
        Computes factual scores for batch of generations using completions_lm.generate

        Args:
            atomic_facts: list of facts for the each generation
            char_level_spans: list of {fact: span} dicts (one list per generation)
            k: number of articles to find as the closest to the query (topic or atomic fact)
            n: number of articles chunks to include in the RAG-prompt
            topics: topics for the each generation

        Returns:
            decisions_by_generation: lists of dicts {atomic_fact, whether it is supported by the RAG-info, char_level span of the fact}
            for the each generation
            passages_for_atoms: dict {fact: list of passages for the fact}
        """
        all_facts, all_topics = [], []
        fact_origin = []  # tracks which generation each fact came from
        for gen_idx, facts in enumerate(atomic_facts):
            all_facts.extend(facts)
            if topics is not None:
                all_topics.extend([topics[gen_idx]] * len(facts))
            fact_origin.extend([gen_idx] * len(facts))

        prompts, passages_for_atoms = await self.get_rag_prompts_and_passages(
            all_facts, topics=all_topics if topics is not None else None, k=k, n=n
        )
        if len(prompts) == 0:
            logger.debug(
                "Could not find RAG-info for generations. Please check your embedding API"
            )
            return [], []
        outputs = await self.lm.generate([p[1] for p in prompts])
        if len(outputs) == 0:
            logger.debug(
                "Could not evaluate if generations facts are true. Please check your completion API"
            )
        decisions_by_generation = [[] for _ in range(len(atomic_facts))]
        for i, (output, (fact, _)) in enumerate(zip(outputs, prompts)):
            gen_idx = fact_origin[i]
            char_span = char_level_spans[gen_idx].get(fact, None)
            generated_answer = output.lower()
            if "true" in generated_answer and "false" in generated_answer:
                is_supported = generated_answer.index("true") > generated_answer.index(
                    "false"
                )
            else:
                is_supported = "true" in generated_answer
            decisions_by_generation[gen_idx].append(
                {"atom": fact, "is_supported": is_supported, "span": char_span}
            )
        return decisions_by_generation, passages_for_atoms

    async def get_rag_prompts_and_passages(
        self, atomic_facts: list[str], k: int, n: int, topics: list[str] = None
    ):
        """
        Retrieves the appropriate information from the database and makes RAG-prompt for the each fact

        Args:
            atomic_facts: list of facts the RAG-prompts are needed for
            k: number of articles to find as the closest to the query (topic or fact itself)
            n: number of chunks from the articles to use as the RAG-info
            topics: topics of the facts
        """
        if topics is not None:
            assert len(topics) == len(
                atomic_facts
            ), "topics and atomic facts must have the same length"
        prompts = []
        texts = (
            await self.retrieval.get_texts_for_queries(topics, k)
            if topics is not None
            else await self.retrieval.get_texts_for_queries(atomic_facts, k)
        )
        if len(texts) == 0:
            return texts, []
        passages_for_atoms = {}
        for i, atom in enumerate(atomic_facts):
            atom = atom.strip()
            if topics is not None:
                passages = self.retrieval.get_bm25_passages(
                    topics[i], atom, texts[topics[i]], n=n
                )
                passages_for_atoms[atom] = passages
                definition = f"Task: answer the question about {topics[i]} based on the given context.\n\n"
            else:
                passages = self.retrieval.get_bm25_passages(
                    topics, atom, texts[atom], n=n
                )
                passages_for_atoms[atom] = passages
                definition = "Task: answer the question based on the given context.\n\n"
            context = ""
            for psg in reversed(passages):
                context += f"Title: {psg["title"]}\nText: {psg["text"]}\n\n".replace(
                    "<s>", ""
                ).replace("</s>", "")
            definition += context.strip()
            prompt = f"{definition.strip()}\n\nInput: {atom.strip()} True or False? Answer True if the information is supported by the context above and False otherwise.\nOutput:"
            prompts.append((atom, prompt))
        return prompts, passages_for_atoms
