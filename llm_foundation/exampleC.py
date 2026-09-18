import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          # reads GROQ_API_KEY from ../.env
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

prompt = "Give one plausible root cause for an intermittent 502 in a Kubernetes ingress."

for temp in (0.0, 0.7, 1.3):
    outs = [
        client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=temp,
            max_tokens=150,
            reasoning_effort="low",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
        for _ in range(3)
    ]
    print(f"\n--- temperature={temp} ---")
    for o in outs:
        print(" •", o.strip().replace("\n", " ")[:110])