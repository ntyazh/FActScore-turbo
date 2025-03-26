import pickle
import os
from factscore.api_requests_processor import process_api_requests_from_list


class CompletionsLLM(object):
    def __init__(
            self,
            completions_request_url,
            completions_model_name,
            cache_file='.cache/factscore'):
        self.cache_file = cache_file
        self.completions_request_url, self.model_name = completions_request_url, completions_model_name


    async def generate(self, prompts: list, sample_idx=0):
        assert isinstance(prompts, list), "prompts to the model must be list"
        prompts = [p.strip() for p in prompts]
        if len(prompts) == 0:
            return []
        generated = await self._generate_completions(prompts)
        if len(generated) == 0:
            print("FAILED ANSWERS")
            print(prompts)
            return []
        if isinstance(generated[0], list):
            generated = generated[0]
        return generated
    

    async def _generate_completions(self, messages: list):
        assert isinstance(messages, list), "prompts to the model must be list"
        if len(messages) == 0:
            return []
        messages = list(map(lambda x: {"messages": [
                        {"role": "user", "content": x}], "model": self.model_name}, messages))
        results, failed_results = await process_api_requests_from_list(requests=messages,
                                                                       request_url=self.completions_request_url,
                                                                       api_key=os.environ["COMPLETIONS_API_KEY"],
                                                                       proxy=os.environ["COMPLETIONS_PROXY"] if os.environ["COMPLETIONS_PROXY"] != "None" else None
                                                                       )
        return results
