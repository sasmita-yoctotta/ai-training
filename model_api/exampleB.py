import os, time, random, hashlib, json, logging
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError

log = logging.getLogger("llm")
load_dotenv()          # reads GROQ_API_KEY from ../.env
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,
)

PRICES = {  # USD per 1M tokens; keep this in config, not code — verify at groq.com/pricing
    "openai/gpt-oss-20b": {"in": 0.10, "out": 0.50},
    "openai/gpt-oss-120b": {"in": 0.15, "out": 0.75},
}

@dataclass
class Result:
    text: str
    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    latency_ms: int
    attempts: int

def call(messages, model="openai/gpt-oss-20b", fallback="openai/gpt-oss-120b",
         max_attempts=4, max_input_tokens=100_000, **kw) -> Result:
    prompt_hash = hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()[:12]

    last_err, started = None, time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        use_model = model if attempt < max_attempts else fallback
        try:
            r = client.chat.completions.create(
                model=use_model, messages=messages, **kw
            )
            u = r.usage
            p = PRICES.get(use_model, {"in": 0, "out": 0})
            res = Result(
                text=r.choices[0].message.content or "",
                model=use_model,
                in_tokens=u.prompt_tokens,
                out_tokens=u.completion_tokens,
                cost_usd=(u.prompt_tokens * p["in"] + u.completion_tokens * p["out"]) / 1e6,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
            )
            log.info("llm_call", extra={"prompt_hash": prompt_hash, **res.__dict__})
            return res
        except (RateLimitError, APITimeoutError) as e:
            last_err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise                      # 4xx is our bug; don't retry
            last_err = e
        sleep = min(2 ** attempt, 20) + random.uniform(0, 1)
        log.warning("retrying in %.1fs (attempt %d)", sleep, attempt)
        time.sleep(sleep)

    raise RuntimeError(f"all {max_attempts} attempts failed") from last_err