
"""
L7 — Course Content Search
==========================
Run with:
    python3 course_search.py

Interactive semantic search over course-summary sentences.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ── Knowledge Base ──────────────────────────────────────────────────────────

sentences = [
    "HTML uses tags to define the structure of a web page",
    "FastAPI automatically validates request data using Pydantic models",
    "Streamlit re-runs the entire script when any widget is interacted with",
    "Python uses names that are snake cased",
    "CSS code is used to make web pages look pretty",
    "HTML puts words and content on a web page",
    "Once you delete something from the terminal it is gone",
    "Always create a virtual environment for Python projects",
    "The touch command creates a file and mkdir creates a directory",
    "You should use rm to remove files from the terminal"
]


# ── Load Model ──────────────────────────────────────────────────────────────

print("Loading model...")

model = SentenceTransformer('all-MiniLM-L6-v2')


# ── Encode Knowledge Base ───────────────────────────────────────────────────

print(f"Embedding {len(sentences)} sentences...")

# Encode all sentences ONCE
doc_embeddings = model.encode(sentences)

print("Document embeddings shape:", doc_embeddings.shape)


# ── Search Loop ─────────────────────────────────────────────────────────────

print("\nSemantic Search — type 'quit' to exit\n")

while True:

    # Ask the user for a search query
    query = input("Search: ").strip()

    # Exit if user types quit
    if query.lower() == "quit":
        print("Goodbye!")
        break

    # Ignore empty searches
    if not query:
        continue

    # Encode the user's query
    query_embedding = model.encode([query])

    print("Query embedding shape:", query_embedding.shape)

    # Calculate cosine similarity
    scores = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    # Get the indices of the 3 highest scores
    top_indices = np.argsort(scores)[::-1][:3]

    # Display the top 3 results
    print("\nTop 3 results:")

    for rank, idx in enumerate(top_indices, 1):

        print(
            f"  {rank}. [{scores[idx]:.4f}] "
            f"{sentences[idx]}"
        )

    print()