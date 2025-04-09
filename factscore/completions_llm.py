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
        self.cache_dict = self.load_cache()
        self.completions_request_url, self.model_name = completions_request_url, completions_model_name


    async def generate(self, prompts: list, sample_idx=0):
        assert isinstance(prompts, list), "prompts to the model must be list"
        prompts = [p.strip() for p in prompts]
        if len(prompts) == 0:
            return []
        # non_cached_pos, out = await self.find_non_cached_generations(prompts, sample_idx)
        # if len(non_cached_pos) == 0:
        #     if isinstance(out[0], list):
        #         out = out[0]
        #     return out

        # prompts = [prompts[j] for j in non_cached_pos]
        generated = await self._generate_completions(prompts)
        # assert len(generated) == len(prompts), f"Generated: {len(generated)}, Propmpts: {len(prompts)}"
        # for i, pos in enumerate(non_cached_pos):
        #     out[pos] = generated[i]
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
        if len(results) == 0:
            print("FAILED RESULTS")
            print(failed_results)
        return results

    async def find_non_cached_generations(self, prompts, sample_idx):
        '''
        if we have already send a prompt to the model and its generation is in our cache,
        we'll pull it from the cache and won't send the request again

        returns
        non_cached_pos: indices of the non-cached generations
        out: list that is filled with the cached generations
        (in the generation time it will be refilled with the non-cached generations)
        '''
        cache_keys = []
        out = [None for _ in prompts]
        for i, p in enumerate(prompts):
            cache_key = f"{p}_{sample_idx}"
            if cache_key in self.cache_dict:
                out[i] = self.cache_dict[cache_key]
            else:
                cache_keys.append(cache_key)
        non_cached_pos = [i for i, g in enumerate(out) if g is None]
        return non_cached_pos, out


    def save_cache(self):
        # load the latest cache first, since if there were other processes
        # running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_dict, f)
        except BaseException:
            pass

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception as e:
                    if not allow_retry:
                        assert False
                    return {}
        else:
            cache = {}
        return cache
