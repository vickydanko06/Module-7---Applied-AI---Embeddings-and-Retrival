"""
L7 — Personal Knowledge Base  (STARTER)
========================================
Run with:
    python starter.py

Your goal: build a persistent ChromaDB knowledge base from course content and
demonstrate metadata-filtered semantic search.

Required features:
    1. A persistent ChromaDB client (data survives restarts)
    2. A collection called "my_knowledge" created with get_or_create_collection
    3. At least 15 documents from at least 3 different modules, each with:
       - A unique ID (e.g. "doc_01")
       - Text content (a sentence or short paragraph)
       - Metadata with at least `module` (e.g. "5") and `topic`
         (e.g. "api", "database", "frontend", "ai")
    4. A search() function that accepts a query and optional module filter
       and returns the top 5 results with distances and metadata
    5. At least 3 test queries:
       - One broad (no filter)
       - One filtered to a specific module
       - One where the query words differ from the stored documents

Key concepts:
    Persistent client:
        client = chromadb.PersistentClient(path="./chroma_db")

    Upsert avoids errors on re-runs:
        collection.upsert(ids=[...], documents=[...], metadatas=[...])

    Metadata filtering:
        collection.query(
            query_texts=["your query"],
            n_results=5,
            where={"module": "5"}   # only return docs from module 5
        )

    Result structure:
        results["ids"][0]        # list of IDs
        results["documents"][0]  # list of document strings
        results["metadatas"][0]  # list of metadata dicts
        results["distances"][0]  # list of distances (lower = more similar)
"""

import chromadb

# ── Client & collection ──────────────────────────────────────────────────────
# TODO: Create a PersistentClient with path="./chroma_db"
# TODO: Get or create a collection called "my_knowledge"
client = chromadb.PersistentClient(path="./chroma_db")
my_knowledge = client.get_or_create_collection(name="my_knowledge",
    metadata={"description": "AI Engineering Foundation knowledge content"},
    configuration={"hnsw": {"space": "cosine"}}
)

# ── Documents ────────────────────────────────────────────────────────────────
# TODO: Define three lists that match in length:
#   ids       — unique strings like "doc_01", "doc_02", ...
#   documents — text summaries from at least 3 different modules
#   metadatas — dicts with at least {"module": "X", "topic": "Y"}
#
# Aim for at least 15 documents.



documents = [
    "FastAPI is a modern Python web framework that automatically validates request data using Pydantic models and generates interactive API documentation.",
    "SQLAlchemy is an ORM that maps Python classes to database tables. It supports relationships like one-to-many and many-to-many.",
    "Streamlit re-runs the entire Python script every time a user interacts with a widget. Use st.session_state to persist data across re-runs.",
    "JWT tokens provide stateless authentication for REST APIs. The token contains encoded user information and is verified on each request.",
    "CSS Flexbox arranges child elements in a row or column. Use display:flex on the container and gap for spacing between items.",
    "The DOM is the browser's tree representation of an HTML page. JavaScript uses querySelector and addEventListener to interact with it.",
    "Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together.",
    "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction (same meaning), 0.0 means unrelated.",
    "REST API design uses predictable, lowercase URIs with plural nouns to represent resources and collections, nested paths for related resources, query parameters for filtering, and stateless requests to improve scalability and consistency.",
    "API documentation explains endpoints, authentication, parameters, request and response formats, status codes, and rate limits, while OpenAPI/Swagger provides interactive documentation and versioning allows APIs to evolve without breaking existing applications.",
    "PUT replaces an entire resource with a complete update, while PATCH modifies only specified fields without changing the remaining data.",
    "API security requires hiding stack traces, using HTTPS, validating inputs, hashing passwords, and setting token expiration to protect sensitive data and prevent unauthorized access.",
    "Streamlit re-runs the entire Python script from top to bottom whenever a user interacts with a widget, causing ordinary variables to reset to their initial values.",
    "Streamlit's st.form() batches multiple user inputs and processes them only when the user clicks submit, preventing unnecessary script reruns and expensive API calls.",
    "ChromaDB organizes text into collections containing documents, each with a unique ID, vector embeddings for semantic search, and optional metadata for filtering results.",
]
ids = [f"doc_{i:02d}" for i in range(1, len(documents) + 1)]

metadatas = [
    {"module": "5", "topic": "api"},
    {"module": "3", "topic": "database"},
    {"module": "6", "topic": "frontend"},
    {"module": "5", "topic": "security"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
    {"module": "4", "topic": "api design"},
    {"module": "4", "topic": "api"},
    {"module": "5", "topic": "api"},
    {"module": "5", "topic": "security"},
    {"module": "6", "topic": "streamlit"},
    {"module": "6", "topic": "streamlit"},
    {"module": "7", "topic": "database"},
]

# TODO: Upsert all documents into the collection

my_knowledge.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
# ── Search function ──────────────────────────────────────────────────────────
def search(query: str, module_filter: str = None, n_results: int = 5):
    """Return top results for query, optionally filtered to a specific module."""

    # Build optional metadata filter
    where = {}
    if module_filter:
        where = {"module": module_filter}

    # Search the collection
    results = my_knowledge.query(
        query_texts=[query],
        n_results=n_results,
        where=where if where else None
    )

    # Print results
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nRank: {i + 1}")
        print(f"ID: {results['ids'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"Document: {doc}")

    return results


# ── Test queries ─────────────────────────────────────────────────────────────
print("=== Broad search (no filter) ===")
search("How do APIs work?")

print("\n=== Filtered to a specific module ===")
# TODO: Call search() with a query and a module_filter
search("How does authentication work?", module_filter="5")

print("\n=== Different words than stored documents ===")
search("How can I prevent users from submitting invalid information?")