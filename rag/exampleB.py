# Index: small child chunks for precision, keep parent text for context.
# Tables: chunks(id, parent_id, content, embedding), parents(id, content)

def retrieve_with_parents(conn, qv, limit=5):
    with conn.cursor() as cur:
        cur.execute("""
            WITH hits AS (
                SELECT DISTINCT ON (parent_id)
                       parent_id, id AS child_id,
                       embedding <=> %s AS dist
                FROM chunks
                ORDER BY parent_id, embedding <=> %s
            )
            SELECT p.id, p.content, h.child_id, h.dist
            FROM hits h JOIN parents p ON p.id = h.parent_id
            ORDER BY h.dist LIMIT %s
        """, (qv, qv, limit))
        return cur.fetchall()