import weaviate
import weaviate.classes as wvc

wc = weaviate.connect_to_local()

wc.collections.create(
    "Chunk",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),   # we supply vectors
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        distance_metric=wvc.config.VectorDistances.COSINE,
        ef_construction=128, max_connections=16,
        quantizer=wvc.config.Configure.VectorIndex.Quantizer.sq(),
    ),
    multi_tenancy_config=wvc.config.Configure.multi_tenancy(enabled=True),
    properties=[
        wvc.config.Property(name="content",    data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="breadcrumb", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="doc_id",     data_type=wvc.config.DataType.TEXT,
                            index_filterable=True, index_searchable=False),
    ],
)

col = wc.collections.get("Chunk")
col.tenants.create([wvc.tenants.Tenant(name="acme")])
acme = col.with_tenant("acme")

with acme.batch.dynamic() as batch:
    for text, bc, doc, vec in rows:
        batch.add_object(
            properties={"content": text, "breadcrumb": bc, "doc_id": doc},
            vector=vec,
        )

res = acme.query.hybrid(
    query="indemnity exclusions for consequential damages",
    vector=query_vector,
    alpha=0.6,                                    # 1.0 = pure vector, 0.0 = pure BM25
    limit=10,
    return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
)
for o in res.objects:
    print(round(o.metadata.score, 4), o.properties["breadcrumb"])

wc.close()