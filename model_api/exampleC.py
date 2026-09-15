from xai_sdk import Client
from xai_sdk.chat import user

client = Client()

chat = client.chat.create(model="grok-4.6")
chat.append(user("Explain idempotency keys in 4 sentences."))

response = None
for response, chunk in chat.stream():
    if chunk.content:
        print(chunk.content, end="", flush=True)
u = response.usage
print(f"\n\n[{u.prompt_tokens} in / {u.completion_tokens} out]")