import numpy as np
from openai import OpenAI

client = OpenAI()
MODEL = "text-embedding-3-small"

texts = [
    "What is our refund window?",
    "Customers may return goods within 30 days of delivery.",
    "Customers may not return goods after delivery.",
    "Invoice INV-88214 is overdue.",
    "Invoice INV-88215 is overdue.",
    "The quarterly revenue forecast was revised upward.",
]

resp = client.embeddings.create(model=MODEL, input=texts)
E = np.array([d.embedding for d in resp.data])
E = E / np.linalg.norm(E, axis=1, keepdims=True)   # normalise, then dot == cosine

S = E @ E.T
np.set_printoptions(precision=3, suppress=True)
print(S)

print("\nquestion vs affirmative answer:", round(float(S[0, 1]), 3))
print("question vs negated answer:   ", round(float(S[0, 2]), 3))
print("INV-88214 vs INV-88215:       ", round(float(S[3, 4]), 3))