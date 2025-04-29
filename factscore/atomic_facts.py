import re

import yaml
from loguru import logger
from nltk.tokenize import sent_tokenize

from factscore.completions_llm import CompletionsLLM


class AtomicFactGenerator:
    def __init__(self, model_name, sentence_level=False):
        """
        Asynchronously breaks list of the generations into lists of facts

        Args:
            model_name: model to use for the factual breakdown
            sentence_level: whether to break down the generation into sentences before the factual breakdown
            as in the original factscore
        """
        self.model_name = model_name
        self.sentence_level = sentence_level
        self.completions_lm = CompletionsLLM(completions_model_name=self.model_name)
        with open("factscore/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
            if self.sentence_level:
                self.prompt = prompts["sentence_level"]
            else:
                self.prompt = prompts["generation_level"]

    async def run(self, generations: list[str]):
        """
        1. Breaks down each generation into list of passages (sentences or generation itself depending on self.sentence_level)
        2. Breaks them down into independent facts with get_atomic_facts

        Returns:
            generation_triplets: list of triplets (passage from the generation, facts from passage, spans of the facts)
            for the each generation
        """
        if isinstance(generations, str):
            generations = [generations]

        all_passages = (
            []
        )  # all_passages[i] == list of the passage from the i-th generation
        generation_ids = []  # maps each passage to its original generation index

        for gen_idx, generation in enumerate(generations):
            if self.sentence_level:
                passages = sent_tokenize(generation)
                all_passages.extend(passages)
                generation_ids.extend([gen_idx] * len(passages))
            else:
                all_passages.append(generation)
                generation_ids.append(gen_idx)

        atoms = await self.get_atomic_facts(all_passages)
        if len(atoms) == 0:
            logger.debug(
                "Could not extract atomic facts from generations. Please check your completion API"
            )
            return []
        results_by_generation = [
            [] for _ in range(len(generations))
        ]  # results_by_generation[i] == \
        # list of tuples (passage, list of the facts from the passage) for the all passages from the i-th generation
        for (sentence, facts), gen_idx in zip(atoms.items(), generation_ids):
            results_by_generation[gen_idx].append((sentence, facts))

        generations_triplets = []
        for gen_idx, atomic_facts_pairs in enumerate(results_by_generation):
            atomic_facts_triplets = []
            for pair in atomic_facts_pairs:
                if not pair[1]:  # if facts for the passage pair[0] were not found
                    atomic_facts_triplets.append((pair[0], [], []))
                    continue
                facts, spans = self.find_facts_spans(generations[gen_idx], pair[1])
                atomic_facts_triplets.append((pair[0], facts, spans))
            generations_triplets.append(atomic_facts_triplets)
        return generations_triplets

    async def get_atomic_facts(self, passages: list):
        """
        Breaks down the passages into independent facts using completions_llm
        Note:
        Each fact looks like <independent self-consistent fact> [[<citation from the passage where the fact is provided>]]
        This is needed for the further fact span search.

        Returns:
            sent_to_facts: dict {passage: list of facts from the passage}
        """
        prompts = [
            self.prompt + f'\nInput Sentence: "{sentence}"\nOutput:'
            for sentence in passages
        ]
        outputs = await self.completions_lm.generate(prompts)
        sent_to_facts = {}
        for sentence, output in zip(passages, outputs):
            sent_to_facts[sentence] = (
                [] if "No facts to extract" in output else self.text_to_facts(output)
            )
        return sent_to_facts

    def text_to_facts(self, llm_output: str):
        """
        Breaks the LLM output into the facts and remove from them any LLM notes
        """
        facts = llm_output.split("- ")[1:]
        facts = [
            (
                sent.strip()[:-1]
                if len(sent) > 0 and sent.strip()[-1] == "\n"
                else sent.strip()
            )
            for sent in facts
        ]
        facts = [re.sub(r"\n\n.*", "", fact, flags=re.DOTALL).strip() for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != ".":
                facts[-1] = facts[-1] + "."
        return facts

    def find_facts_spans(self, generation: str, facts: list[str]):
        """
        1. Extracts citations of the facts from the generation
        2. Uses these citations to find the facts spans

        Returns
            facts: the facts without citations
            spans: the char-level spans of the facts (indices where a fact begins and where it ends in the generation)
        """
        postprocessed_facts, spans = [], []
        pattern = r"\[\[(.*?)\]\]"
        generation = generation.lower()
        for fact in facts:
            try:
                citation = re.findall(pattern, fact)[0]
            except IndexError:
                logger.warning(
                    "Couldn't find the pattern [[ ]] for the fact spans in the generation:",
                    fact,
                )
                continue
            start_citation_index = generation.find(citation.lower())
            if start_citation_index == -1:
                span = self._find_maximal_substring_with_span(
                    generation, citation.lower()
                )
                if span[0] == 0 and span[1] == 0:
                    logger.warning(
                        f"Couldn't find the citation in the generation.\nCitation: {citation}\nGeneration: {generation}"
                    )
                    continue
                postprocessed_facts.append(re.sub(pattern, "", fact).strip())
                spans.append(span)
                continue
            end_citation_index = start_citation_index + len(citation)
            postprocessed_facts.append(re.sub(pattern, "", fact).strip())
            spans.append((start_citation_index, end_citation_index))
        return postprocessed_facts, spans

    def _find_maximal_substring_with_span(self, main_str, sub_str):
        sub_words = sub_str.split()
        max_len = len(sub_words)
        end = len(sub_words)
        for length in range(max_len, 0, -1):
            candidate = " ".join(sub_words[end - length : end])
            if candidate in main_str:
                start_idx = main_str.find(candidate)
                end_idx = start_idx + len(candidate)
                return (start_idx, end_idx)
        return (0, 0)
