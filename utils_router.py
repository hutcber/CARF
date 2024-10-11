import os
import sys
import openai
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["openai/gpt-4", "openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-instruct", 'qwen-turbo', 'qwen-plus', 'qwen-max']

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def router_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    # os.environ["http_proxy"] = "127.0.0.1:7890"
    # os.environ["https_proxy"] = "127.0.0.1:7890"
    # gets API Key from environment variable OPENAI_API_KEY
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENAI_API_KEY')
    )

    response = client.completions.create(
        model='openai/gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def router_chat(prompt: str, model: Model, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> str:
    assert model != "text-davinci-003"
    # os.environ["http_proxy"] = "127.0.0.1:7890"
    # os.environ["https_proxy"] = "127.0.0.1:7890"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # gets API Key from environment variable OPENAI_API_KEY
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENAI_API_KEY')
    )

    response = client.chat.completions.create(
        model = model,
        messages = messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    r = router_chat("say hello world", 'openai/gpt-3.5-turbo')
    print(r)