"""
L7 — Evaluate Your Search System  (STARTER)
=============================================
Run with:
    python starter.py

Your goal: build an evaluation framework that measures precision and recall
for a semantic search system across multiple settings.

Required features:
    1. A ChromaDB collection with at least 12 documents across 3–4 topics
    2. An evaluation set: at least 6 test queries, each with a list of
       expected relevant document IDs
    3. An evaluate() function that computes per-query and average
       precision & recall
    4. Results at 3 different settings (vary threshold, n_results, or both)
    5. A written analysis in the output

Key concepts:
    Precision — of the results returned, how many were relevant?
        precision = len(relevant ∩ retrieved) / len(retrieved)

    Recall — of all relevant documents, how many did we retrieve?
        recall = len(relevant ∩ retrieved) / len(relevant)

    ChromaDB returns distances (lower = more similar).
    Convert to similarity for thresholding:
        similarity = 1 - distance   (for cosine distance)
        OR just filter by distance <= threshold

    Example evaluation set entry:
        {
            "query": "How does FastAPI validate request data?",
            "relevant_ids": ["doc_03", "doc_07"]
        }
"""

import chromadb

# ── Documents ─────────────────────────────────────────────────────────────────
# TODO: Define ids, documents, and metadatas lists (at least 12 items)
#       Cover 3–4 topics so that some queries are easy and some are hard.
#       Use unique IDs like "doc_01", "doc_02", ... — these are what the
#       evaluation set references.



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


# ── ChromaDB setup ────────────────────────────────────────────────────────────
# TODO: Create a chromadb.EphemeralClient() (in-memory, no files needed)
client = chromadb.EphemeralClient()
# TODO: Create or get a collection called "eval_collection"
collection = client.get_or_create_collection(
    name="eval_collection"
)
# TODO: Upsert all documents
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"Successfully added {collection.count()} documents to ChromaDB!")

# ── Evaluation set ────────────────────────────────────────────────────────────
# TODO: Define eval_set — a list of dicts, each with:
#   "query":        the natural language query string
#   "relevant_ids": list of document IDs that are truly relevant

# ── Evaluation set ────────────────────────────────────────────────────────────

eval_set = [
    {
        "query": "How do I authenticate API requests?",
        "relevant_ids": ["doc_04", "doc_12"]
    },
    {
        "query": "How does Streamlit remember data between interactions?",
        "relevant_ids": ["doc_03", "doc_13"]
    },
    {
        "query": "What is semantic similarity and how is it measured?",
        "relevant_ids": ["doc_07", "doc_08"]
    },
    {
        "query": "How does ChromaDB organize documents for semantic search?",
        "relevant_ids": ["doc_15", "doc_07"]
    },
    {
        "query": "How do I make a webpage interactive?",
        "relevant_ids": ["doc_05", "doc_06"]
    },
    {
        "query": "What is the difference between PUT and PATCH?",
        "relevant_ids": ["doc_11"]
    },
]



# ── Evaluation function ────────────────────────────────────────────────────────

def evaluate(n_results: int, distance_threshold: float = None):
    """
    Run every query in eval_set, compute precision and recall for each,
    and print per-query results plus averages.
    """

    print(f"\n=== Evaluation: n_results={n_results}, threshold={distance_threshold} ===")

    precisions = []
    recalls = []

    for i, entry in enumerate(eval_set, 1):
        query = entry["query"]
        relevant_ids = set(entry["relevant_ids"])

        # Query the ChromaDB collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances"]
        )

        # Extract retrieved document IDs
        retrieved_ids = results["ids"][0]
        distances = results["distances"][0]

        # Apply distance threshold if one is provided
        if distance_threshold is not None:
            retrieved_ids = [
                doc_id
                for doc_id, distance in zip(retrieved_ids, distances)
                if distance <= distance_threshold
            ]

        # Convert retrieved IDs into a set
        retrieved_set = set(retrieved_ids)

        # Calculate how many retrieved documents are relevant
        overlap = len(relevant_ids.intersection(retrieved_set))

        # Calculate precision
        precision = (
            overlap / len(retrieved_set)
            if retrieved_set else 0.0
        )

        # Calculate recall
        recall = (
            overlap / len(relevant_ids)
            if relevant_ids else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)

        print(
            f"  Query {i}: P={precision*100:.1f}%  "
            f"R={recall*100:.1f}%  | '{query[:50]}'"
        )

    # Calculate average precision and recall
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0

    print(f"  AVERAGE: P={avg_p*100:.1f}%  R={avg_r*100:.1f}%")

    return avg_p, avg_r

# ── Run at 3 settings ─────────────────────────────────────────────────────────
# TODO: Call evaluate() with 3 different combinations of n_results / threshold
#       e.g. evaluate(n_results=3)
#            evaluate(n_results=5)
#            evaluate(n_results=5, distance_threshold=0.5)
print("\n\n========== COMPARING SEARCH SETTINGS ==========")

# Setting 1: Retrieve 3 documents per query
evaluate(n_results=3)

# Setting 2: Retrieve 5 documents per query
evaluate(n_results=5)

# Setting 3: Retrieve 5 documents, but filter by distance
evaluate(n_results=5, distance_threshold=0.5)


# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n=== ANALYSIS ===")
# TODO: Print your observations as a multi-line string or separate print() calls.
#       Answer:
#         - Which queries consistently performed well? Why?
#         - Which queries failed? What caused the low score?
#         - What would you change to improve the weakest queries?
#         - How did increasing n_results affect precision vs recall?
print("\n=== ANALYSIS ===")

print("""
1. Which queries consistently performed well? Why?

Queries about Streamlit, semantic similarity, and PUT vs PATCH
performed well because the documents contain specific keywords
and information that closely match the search questions.

2. Which queries failed? What caused the low score?

Some queries may have performed poorly because multiple documents
contain similar information. For example, API authentication
overlaps with API security. This can make it harder for the search
system to identify the most relevant document.

3. What would you change to improve the weakest queries?

I would improve the documents by adding more specific information,
using clearer descriptions, and reducing repeated content.
I would also experiment with different distance thresholds
and search settings to improve the results.

4. How did increasing n_results affect precision vs recall?

Increasing n_results from 3 to 5 allows the search system to
retrieve more documents. This can improve recall because more
relevant documents may be found. However, precision may decrease
because additional irrelevant documents can also be returned.

Overall, the best search setting depends on whether precision
or recall is more important for the application.
""")