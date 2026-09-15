import time, random, hashlib, json, logging
from dataclasses import dataclass
import grpc
from xai_sdk import Client

log = logging.getLogger("llm")
client = Client(timeout=30.0)

# gRPC codes that mean "our request was wrong" — retrying won't help.
_NON_RETRYABLE = {
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.UNAUTHENTICATED,
    grpc.StatusCode.PERMISSION_DENIED,
    grpc.StatusCode.NOT_FOUND,
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

def call(messages, model="grok-4.6", fallback="grok-4-fast",
         max_attempts=4, max_input_tokens=100_000, **kw) -> Result:
    prompt_hash = hashlib.sha256(
        json.dumps([str(m) for m in messages], sort_keys=True).encode()
    ).hexdigest()[:12]

    last_err, started = None, time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        use_model = model if attempt < max_attempts else fallback
        try:
            chat = client.chat.create(model=use_model, **kw)
            for m in messages:
                chat.append(m)
            r = chat.sample()
            u = r.usage
            res = Result(
                text=r.content or "",
                model=use_model,
                in_tokens=u.prompt_tokens,
                out_tokens=u.completion_tokens,
                cost_usd=r.cost_usd or 0.0,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
            )
            log.info("llm_call", extra={"prompt_hash": prompt_hash, **res.__dict__})
            return res
        except grpc.RpcError as e:
            if e.code() in _NON_RETRYABLE:
                raise                      # our bug; don't retry
            last_err = e
        sleep = min(2 ** attempt, 20) + random.uniform(0, 1)
        log.warning("retrying in %.1fs (attempt %d)", sleep, attempt)
        time.sleep(sleep)

    raise RuntimeError(f"all {max_attempts} attempts failed") from last_err