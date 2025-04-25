import os

from factscore.api_requests_processor import process_api_requests_from_list


class CompletionsLLM:
    def __init__(
            self,
            completions_model_name: str
    ):
        '''
        Asynchronously sends requests to the completion llm

        Args:
            completions_model_name: (for example, gpt-4o-mini)
        '''
        self.model_name = completions_model_name

    async def generate(self, prompts: list):
        '''
        Transforms prompts into the openai-compatible request jsons and sends them to os.environ["COMPLETIONS_BASE_URL"]
        '''
        assert isinstance(prompts, list), "prompts to the model must be list"
        prompts = [p.strip() for p in prompts]
        if len(prompts) == 0:
            return []
        messages = list(map(lambda x: {"messages": [{"role": "user", "content": x}],
                                       "model": self.model_name},
                            prompts))
        responses = await process_api_requests_from_list(requests=messages,
                                                         request_url=os.environ["COMPLETIONS_BASE_URL"],
                                                         api_key=os.environ["COMPLETIONS_API_KEY"],
                                                         proxy=os.environ["COMPLETIONS_PROXY"] if os.environ["COMPLETIONS_PROXY"] != "None" else None
                                                         )
        if len(responses) > 0 and isinstance(responses[0], list):
            responses = responses[0]
        return responses
    