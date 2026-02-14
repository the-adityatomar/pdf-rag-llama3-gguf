def retrieve_context(index, query, top_k=3):
    """
    Retrieves top-k relevant chunks from index.
    """

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    response = query_engine.query(query)

    source_nodes = response.source_nodes

    results = []

    for node in source_nodes:
        results.append({
            "text": node.node.get_content(),
            "score": node.score
        })

    return results
