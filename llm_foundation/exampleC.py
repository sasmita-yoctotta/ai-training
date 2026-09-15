import os

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user

load_dotenv()          # reads XAI_API_KEY from ../.env
client = Client(api_key=os.environ["XAI_API_KEY"])

prompt = "Give one plausible root cause for an intermittent 502 in a Kubernetes ingress."

for temp in (0.0, 0.7, 1.3):
    outs = []
    for _ in range(3):
        chat = client.chat.create(model="grok-4.6", temperature=temp, max_tokens=40)
        chat.append(user(prompt))
        outs.append(chat.sample().content)
    print(f"\n--- temperature={temp} ---")
    for o in outs:
        print(" •", o.strip().replace("\n", " ")[:110])
