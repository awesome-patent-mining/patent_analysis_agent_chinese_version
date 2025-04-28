import asyncio
import warnings

import litellm

# Filter out the Pydantic warning
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:*")

from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()

DEFAULT_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
litellm.set_verbose = True

class LLM:

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    async def completion(self, prompt_messages: str, **kwargs):
        response = await acompletion(
            model=self.model, messages=prompt_messages, **kwargs
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    llm = LLM()
    print(asyncio.run(llm.completion("What is the capital of France?")))
