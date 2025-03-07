import numpy as np
import re
import string
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from factscore.completions_llm import CompletionsLLM
nltk.download("punkt")


SENTENCE_INSTRUCT_PROMPT = """
Task: Given the following sentence, break it into individual, independent facts with exact citations pointing to the relevant portion of the original sentence.
Do not change anything in the citations.

Example 1:
Input Sentence: "Albert Einstein developed the theory of relativity, which revolutionized modern physics."
Output:
- Albert Einstein developed the theory of relativity [[Albert Einstein developed the theory of relativity]]
- The theory of relativity revolutionized modern physics [[the theory of relativity, which revolutionized modern physics]]

Example 2:
Input Sentence: "During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence."
Output:
- Turing worked for the Government Code and Cypher School at Bletchley Park [Turing worked for the Government Code and Cypher School at Bletchley Park]
- Bletchley Park was Britain's codebreaking centre [Bletchley Park, Britain's codebreaking centre]
- Bletchley Park produced Ultra intelligence [Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence]

Example 3:
Input Sentence: "He was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer."
Output:
- He was highly influential in the development of theoretical computer science [[He was highly influential in the development of theoretical computer science]]
- He provided a formalisation of the concepts of algorithm and computation with the Turing machine [[providing a formalisation of the concepts of algorithm and computation with the Turing machine]]
- The Turing machine can be considered a model of a general-purpose computer [[the Turing machine, which can be considered a model of a general-purpose computer]]

Example 4:
Input Sentence: "The method is used in radiology, archaeology, biology, atmospheric science and other areas of science."
Output:
- The method is used in radiology [[The method is used in radiology]]
- The method is used in archaeology [[archaeology]]
- The method is used in biology [[biology]]
- The method is used in atmospheric science [[atmospheric science]]
- The method is used in other areas of science [[other areas of science]]
"""

PARAGRAPH_INSTRUCT_PROMPT = """
Task: Given the following passage, break it into individual, independent facts with exact citations pointing to the relevant portion of the original passage.
Do not change anything in the citations.

Example 1:
Input Passage: "Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence. He led Hut 8, the section responsible for German naval cryptanalysis. Turing devised techniques for speeding the breaking of German ciphers, including improvements to the pre-war Polish bomba method, an electromechanical machine that could find settings for the Enigma machine. He played a crucial role in cracking intercepted messages that enabled the Allies to defeat the Axis powers in many engagements, including the Battle of the Atlantic."
Output:
- Turing was highly influential in the development of theoretical computer science [[Turing was highly influential in the development of theoretical computer science]]
- Turing provided a formalisation of the concepts of algorithm and computation with the Turing machine [[providing a formalisation of the concepts of algorithm and computation with the Turing machine]]
- The Turing machine can be considered a model of a general-purpose computer [[the Turing machine, which can be considered a model of a general-purpose computer]]
- Turing worked for the Government Code and Cypher School at Bletchley Park [[Turing worked for the Government Code and Cypher School at Bletchley Park]]
- Bletchley Park was Britain's codebreaking centre [[Bletchley Park, Britain's codebreaking centre]]
- Bletchley Park produced Ultra intelligence [[Bletchley Park, Britain's codebreaking centre that produced Ultra intelligence]]
- Turing led Hut 8 [[He led Hut 8]
- Hut 8 is the section responsible for German naval cryptanalysis [[Hut 8, the section responsible for German naval cryptanalysis]]
- Turing devised techniques for speeding the breaking of German ciphers [[Turing devised techniques for speeding the breaking of German ciphers]]
- Turing devised improvements to the pre-war Polish bomba method [[including improvements to the pre-war Polish bomba method]]
- The pre-war Polish bomba method is an electromechanical machine that could find settings for the Enigma machine. [[the pre-war Polish bomba method, an electromechanical machine that could find settings for the Enigma machine]]
- He played a crucial role in cracking intercepted messages that enabled the Allies to defeat the Axis powers in many engagements, including the Battle of the Atlantic. [[He played a crucial role in cracking intercepted messages that enabled the Allies to defeat the Axis powers in many engagements, including the Battle of the Atlantic.]]

Example 2:
Input Passage: "The method is used in radiology, archaeology, biology, atmospheric science and other areas of science."
Output:
- The method is used in radiology [[The method is used in radiology]]
- The method is used in archaeology [[archaeology]]
- The method is used in biology [[biology]]
- The method is used in atmospheric science [[atmospheric science]]
- The method is used in other areas of science [[other areas of science]]
"""


class AtomicFactGenerator(object):
    def __init__(self, request_url, model_name, sentence_level=False, fact_postprocess=False):
        '''
        request_url: which url to send llm requests to (in our case completions_request_url)
        model_name: what model to use
        fact_postprocess: whether to use the original postprocessing of the facts
        '''
        self.fact_postprocess = fact_postprocess
        self.model_name = model_name
        self.sentence_level = sentence_level
        self.completions_lm = CompletionsLLM(
            completions_model_name=self.model_name,
            completions_request_url=request_url)
        if self.sentence_level:
            self.prompt = SENTENCE_INSTRUCT_PROMPT
        else:
            self.prompt = PARAGRAPH_INSTRUCT_PROMPT

    def save_cache(self):
        self.completions_lm.save_cache()


    async def run(self, generation, cost_estimate=False):
        """
        Convert the generation into a set of atomic facts.
        Return a total words cost if cost_estimate == True.
        """
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [
            para.strip() for para in generation.split("\n") if len(
                para.strip()) > 0]
        
        if self.sentence_level:
            sentences = []
            para_breaks = []
            for para_idx, paragraph in enumerate(paragraphs):
                if para_idx > 0:
                    para_breaks.append(len(sentences))
                initials = detect_initials(paragraph)
                curr_sentences = sent_tokenize(paragraph)
                curr_sentences = fix_sentence_splitter(curr_sentences, initials)
                sentences += curr_sentences
        atoms_or_estimate = await self.get_atomic_facts(sentences if self.sentence_level else paragraphs,
                                                                        cost_estimate=cost_estimate)
        if cost_estimate:
            return atoms_or_estimate
        
        atomic_facts_pairs = [] # pair (orig sentence, [list of facts from it])
        for sentence_or_paragraph, facts in(atoms_or_estimate).items():
            atomic_facts_pairs.append((sentence_or_paragraph, facts))

        if self.fact_postprocess:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(
                atomic_facts_pairs, list(para_breaks))

        atomic_facts_triplets = [] # triplets of the type (sentence/passage, [atomic facts from the sentence], [spans of the facts])
        for pair in atomic_facts_pairs:
            facts, char_level_spans = await self.find_facts_spans(generation, pair[1])
            atomic_facts_triplets.append((pair[0], facts, char_level_spans))
        return atomic_facts_triplets
        

    async def get_atomic_facts(self, sentences: list, cost_estimate=False):
        '''
        get atomic facts from sentences if self.sentence_level else get atomic facts from the whole paragraphs, 
        without breaking them into the facts
        '''
        if cost_estimate:
            prompt_one = [self.prompt + f"""
                       Now process the following passage:\nInput passage: "{sentences[0]}"\nOutput:
                       """]
            encoding = tiktoken.encoding_for_model('gpt-4o-mini')
            input_tokens = encoding.encode(prompt_one)
            return input_tokens * len(sentences)

        prompts = []
        for sentence in sentences:
            prompt = self.prompt + f"""
                       Now process the following passage:\nInput passage: "{sentence}"\nOutput:
                       """
            prompts.append(prompt)

        outputs = await self.completions_lm.generate(prompts)
        sent_to_facts = {}  # dict {sentence: facts from the sentence}
        if outputs is not None:
            for i, output in enumerate(outputs):
                sent_to_facts[sentences[i]] = await self.text_to_facts(output)
            return sent_to_facts
        

    async def text_to_facts(self, text):
        '''
        breaks llm output into facts and remove from them any llm notes
        (sometimes llm returns outputs like "<fact>\n\n<note of the llm>",
        that is inappropriate because we want just the fact without any extra information)
        '''
        facts = text.split("- ")[1:]
        facts = [sent.strip()[:-1] if len(sent) > 0 and sent.strip()[-1]
                 == '\n' else sent.strip() for sent in facts]
        facts = [re.sub(r'\n\n.*', '', fact, flags=re.DOTALL).strip()
                 for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != '.':
                facts[-1] = facts[-1] + '.'
        return facts
    

    async def find_facts_spans(self, original_text: str, generations: list):
        '''
        for each llm generation does the following:
        1. extracts citations of the facts from the generation
        2. using these citations to find the facts spans

        returns
        facts: the facts without citations
        spans: the char-level spans of the facts
        (indices where a fact begins and where it ends in the original text)
        '''
        facts, spans = [], []
        pattern = r'\[\[(.*?)\]\]'
        for generation in generations:
            try:
                citation = re.findall(pattern, generation)[0]
            except IndexError:
                print("couldn't find the pattern [[ ]] for the fact spans in the generation:", generation)
                continue
            start_citation_index = original_text.find(citation)
            if start_citation_index == -1:
                print(
                    f"couldn't find the citation in the original text. citation: {citation}\noriningal text: {original_text}")
                continue
            end_citation_index = start_citation_index + len(citation)
            facts.append(re.sub(pattern, '', generation).strip())
            spans.append((start_citation_index, end_citation_index))
        return facts, spans


'''
the functions bellow are from the original factscore.
i'm not sure how useful they are, but they don't ruin anything so i've decided not to remove them
'''


def postprocess_atomic_facts(_atomic_facts, para_breaks):
    '''
    postprocess_atomic_facts will fix minor issues from InstructGPT
    it is supposed to handle sentence splitter issue too, but since here
    we fixed sentence splitter issue already,
    the new para_breaks should be identical to the original para_breaks
    '''
    verbs = [
        "born.",
        " appointed.",
        " characterized.",
        " described.",
        " known.",
        " member.",
        " advocate.",
        "served.",
        "elected."]
    permitted_verbs = ["founding member."]
    nlp = spacy.load("en_core_web_sm")
    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any(
                    [fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j,
                        other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set(
                [e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities == covered_entities
        except Exception:
            # there is a bug in spacy entity linker, so just go with the
            # previous facts
            new_facts = facts
        new_atomic_facts.append((sent, new_facts))
    return new_atomic_facts, new_para_breaks


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"]
MONTHS = [m.lower() for m in MONTHS]


def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False


def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True


def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    # find all numeric values in the text
    numeric_values = re.findall(pattern, text)
    # convert the values to float and return as a list
    return set([value for value in numeric_values])


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in [
            "DATE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
            "ORDINAL",
                "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)
    return entities


def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False


def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]


def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [
                t.strip() for t in initial.split(".") if len(
                    t.strip()) > 0]
            for i, (sent1, sent2) in enumerate(
                    zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(
                        alpha1 +
                        ".") and sent2.startswith(
                        alpha2 +
                        "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [
                        curr_sentences[i] + " " + curr_sentences[i + 1]] + curr_sentences[i + 2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences
