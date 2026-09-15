import os, psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI

client = OpenAI()
conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
register_vector(conn)

HYBRID_SQL = open("hybrid.sql").read()   # the query above

def embed(texts: list[str]) -> list[list[float]]:
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in r.data]

def index_chunks(tenant_id, rows):       # rows: [(doc_id, breadcrumb, content), ...]
    vecs = embed([f"[{b}]\n{c}" for _, b, c in rows])
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (tenant_id, doc_id, breadcrumb, content,
                                   embedding, embedding_model)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            [(tenant_id, d, b, c, v, "text-embedding-3-small")
             for (d, b, c), v in zip(rows, vecs)],
        )

def search(tenant_id, query, limit=10):
    qv = embed([query])[0]
    with conn.cursor() as cur:
        cur.execute("SET LOCAL app.tenant_id = %s", (tenant_id,))
        cur.execute("SET LOCAL hnsw.ef_search = 100")    # recall dial
        cur.execute(HYBRID_SQL, (qv, query))
        return cur.fetchmany(limit)