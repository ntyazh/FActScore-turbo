import os
from factscore.api_requests_processor import process_api_requests_from_list
from loguru import logger


class CompletionsLLM:
    def __init__(
            self,
            completions_request_url: str,
            completions_model_name: str
    ):
        '''
        Asynchronously sends requests to the completion llm

        Args:
            completions_request_url: url to send the requests to (for example, https://api.openai.com/v1/chat/completions)
            completions_model_name: (for example, gpt-4o-mini)
        '''
        self.completions_request_url, self.model_name = completions_request_url, completions_model_name

    async def generate(self, prompts: list):
        '''
        Transforms prompts into the openai-compatible request jsons and sends them to self.completions_request_url
        '''
        assert isinstance(prompts, list), "prompts to the model must be list"
        prompts = [p.strip() for p in prompts]
        if len(prompts) == 0:
            return []
        messages = list(map(lambda x: {"messages": [{"role": "user", "content": x}],
                                       "model": self.model_name},
                            prompts))
        responses = await process_api_requests_from_list(requests=messages,
                                                         request_url=self.completions_request_url,
                                                         api_key=os.environ["COMPLETIONS_API_KEY"],
                                                         proxy=os.environ["COMPLETIONS_PROXY"] if os.environ["COMPLETIONS_PROXY"] != "None" else None
                                                         )
        if len(responses) > 0 and isinstance(responses[0], list):
            responses = responses[0]
        return responses
    