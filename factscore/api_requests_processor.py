import asyncio

import aiohttp
from loguru import logger


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    request_url: str,
    proxy: str,
    request_header: dict,
    request_json: dict,
    max_retries=5,
    retry_delay=1.0,
):

    for attempt in range(max_retries):
        try:
            async with session.post(
                url=request_url, proxy=proxy, headers=request_header, json=request_json
            ) as response:
                response.raise_for_status()
                response = await response.json()
                if "chat" in request_url:
                    return get_content_message_from_response(response)
                elif "embeddings" in request_url:
                    return get_embedding_from_response(response)
                else:
                    logger.error(f"Invalid request url: {request_url}")
                    raise ValueError(f"Unsupported request url: {request_url}")

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All retries are over for {request_url}")
                raise e
            await asyncio.sleep(retry_delay)


def get_embedding_from_response(response):
    return response["data"][0]["embedding"]


def get_content_message_from_response(response):
    return response["choices"][0]["message"]["content"]


async def process_api_requests_from_list(
    requests: list,
    request_url: str,
    proxy: str,
    api_key: str,
    max_attempts=4,
    retry_delay=2,
):
    request_header = {"Authorization": f"Bearer {api_key}"}
    if proxy is not None and proxy != "None":
        request_header["Proxy-Authorization"] = proxy
    responses_list = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_retries(
                session=session,
                request_url=request_url,
                proxy=proxy,
                request_header=request_header,
                request_json=request_json,
                max_retries=max_attempts,
                retry_delay=retry_delay,
            )
            for request_json in requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result
            responses_list.append(result)
    return responses_list
