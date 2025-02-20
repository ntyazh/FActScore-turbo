import aiohttp
import asyncio


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    request_url: str,
    proxy: str,
    request_header,
    request_json,
    max_retries=5,
    retry_delay=1.0,
    retry_condition=None):
    for attempt in range(max_retries):
        try:
            async with session.post(
                    url=request_url,
                    proxy=proxy,
                    headers=request_header,
                    json=request_json
                    ) as response:
                response.raise_for_status()
                response = await response.json()
                if "chat" in request_url:
                    return get_content_message_from_response(response)
                elif "embeddings" in request_url:
                    return get_embedding_from_response(response)
                return
                        
        except Exception as e:
            print("failed with exception", e)
            print(request_json)
            if retry_condition and not retry_condition(e):
                raise  
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise  
    return None


def get_embedding_from_response(response):
    try:
        return response['data'][0]['embedding']
    except KeyError:
        return None


def get_content_message_from_response(response):
    return response['choices'][0]['message']['content'] #chat


async def process_api_requests_from_list(
        requests: list,
        request_url: str,
        proxy: str,
        api_key: str,
        max_attempts=4,
):
    """
    processes API requests in parallel, throttling to stay under rate limits.
    """
    seconds_to_pause_after_error = 2
    seconds_to_sleep_each_loop = 0.003 # 1 ms limits max throughput to 1,000 requests per second

    if proxy is None or proxy == "None":
        request_header = {"Authorization": f"Bearer {api_key}",
                        #   "Proxy-Authorization": proxy
                        }
    else:
        request_header = {"Authorization": f"Bearer {api_key}",
                          "Proxy-Authorization": proxy
                        }
    responses_list, failed_results = [], []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_retries(session=session, 
                                    request_url=request_url, 
                                    proxy=proxy,
                                    request_header=request_header, 
                                    request_json=request_json,
                                    max_retries=max_attempts,
                                    retry_delay=seconds_to_pause_after_error) 
                                    for request_json in requests
                                    ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(result)
            else:
                responses_list.append(result)
    return responses_list, failed_results
