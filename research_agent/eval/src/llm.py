import asyncio

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

DEFAULT_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"


class LLM:

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    async def completion(self, prompt: str, **kwargs):
        response = completion(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    llm = LLM()
    print(asyncio.run(llm.completion("What is the capital of France?")))
