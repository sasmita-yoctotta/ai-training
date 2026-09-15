import json, statistics
from xai_sdk import Client
from xai_sdk.chat import system, user
client = Client()

JUDGE = """You grade an answer against reference material. Return JSON only:
{"faithful": true|false, "faithful_reason": "...",
 "correct": true|false, "correct_reason": "..."}

faithful = every claim in the answer is supported by the CONTEXT.
correct  = the answer matches the REFERENCE ANSWER in substance."""

def judge(question, context, answer_text, reference):
    chat = client.chat.create(model="grok-4.6", temperature=0, response_format="json_object")
    chat.append(system(JUDGE))
    chat.append(user(
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer_text}\n\nREFERENCE ANSWER:\n{reference}"
    ))
    return json.loads(chat.sample().content)

def evaluate(golden, pipeline):
    """golden: [{question, reference, gold_chunk_ids}]"""
    rows = []
    for g in golden:
        out = pipeline(g["question"])
        gold = set(g["gold_chunk_ids"])
        got  = set(out["chunks"])
        verdict = judge(g["question"], out.get("context", ""),
                        out["answer"], g["reference"])
        rows.append({
            "question": g["question"],
            "recall": len(gold & got) / max(len(gold), 1),
            "hit": bool(gold & got),
            **verdict,
            "bad_citations": len(out["hallucinated_citations"]),
        })

    print(f"retrieval hit rate : {statistics.mean(r['hit'] for r in rows):.1%}")
    print(f"mean recall        : {statistics.mean(r['recall'] for r in rows):.1%}")
    print(f"faithfulness       : {statistics.mean(r['faithful'] for r in rows):.1%}")
    print(f"correctness        : {statistics.mean(r['correct'] for r in rows):.1%}")
    print(f"bad citations      : {sum(r['bad_citations'] for r in rows)}")

    # the diagnostic that matters: retrieved correctly but answered wrong
    gen_fail = [r for r in rows if r["hit"] and not r["correct"]]
    print(f"\ngeneration failures despite good retrieval: {len(gen_fail)}")
    for r in gen_fail[:5]:
        print("  -", r["question"], "→", r["correct_reason"][:80])
    return rows