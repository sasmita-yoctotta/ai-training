import json, os, statistics

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          # reads GROQ_API_KEY from ../.env
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

JUDGE = """You grade an answer against reference material. Return JSON only:
{"faithful": true|false, "faithful_reason": "...",
 "correct": true|false, "correct_reason": "..."}

faithful = every claim in the answer is supported by the CONTEXT.
correct  = the answer matches the REFERENCE ANSWER in substance."""

def judge(question, context, answer_text, reference):
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b", temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                f"ANSWER:\n{answer_text}\n\nREFERENCE ANSWER:\n{reference}"},
        ],
    )
    return json.loads(r.choices[0].message.content)

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