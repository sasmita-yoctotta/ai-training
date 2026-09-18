import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          # reads GROQ_API_KEY from ../.env
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

stream = client.chat.completions.create(
   model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain idempotency keys in 4 sentences."}],
    stream=True,
    stream_options={"include_usage": True},
)
usage = None
for chunk in stream:
    if chunk.usage:
        usage = chunk.usage
    if chunk.choices and (d := chunk.choices[0].delta.content):
        print(d, end="", flush=True)
print(f"\n\n[{usage.prompt_tokens} in / {usage.completion_tokens} out]")