from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            notifications_min_severity="OFF",
        )
    return _driver


def run_query(query, params=None):
    with get_driver().session() as session:
        return [record.data() for record in session.run(query, params or {})]


def setup_constraints():
    statements = [
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT doc_url IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE",
    ]
    for stmt in statements:
        run_query(stmt)


def setup_vector_index(dimensions=384):
    run_query(
        """
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: %d,
            `vector.similarity_function`: 'cosine'
        }}
        """ % dimensions
    )


def close():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


if __name__ == "__main__":
    setup_constraints()
    print(run_query("SHOW CONSTRAINTS YIELD name RETURN name"))
    close()