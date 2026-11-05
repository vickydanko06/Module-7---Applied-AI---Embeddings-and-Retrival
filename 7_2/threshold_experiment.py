"""
L7 — Similarity Threshold Experiment  (STARTER)
================================================
Run with:
    python starter.py

Your goal: show how different similarity thresholds change which results are
returned for the same query.

Required features:
    1. A `knowledge_base` list of at least 15 sentences across 3–4 distinct
       topics (e.g. Python, cooking, space, music)
    2. A `test_queries` list of at least 5 queries — some clearly on one topic,
       some ambiguous
    3. Embed all knowledge-base sentences once
    4. For each query, compute similarity scores against all documents
    5. Display results at three thresholds: 0.3, 0.5, and 0.7
       For each threshold show: count of results, which sentences pass,
       and a note about any relevant sentences that are missed

Key concepts:
    Filtering by threshold:
        passing = [(score, sentence)
                   for score, sentence in zip(scores, knowledge_base)
                   if score >= threshold]
        passing.sort(reverse=True)

    Expected output format:
        Query: "How do I deploy my Python app?"

          Threshold 0.3: 6 results
            [0.72] Deploy FastAPI with uvicorn and Docker
            ...
          Threshold 0.5: 3 results
            ...
          Threshold 0.7: 1 result
            ...
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ───────────────────────────────────────────────────────────
# TODO: Create `knowledge_base` — at least 15 sentences across 3–4 topics.
#       Aim for roughly equal coverage of each topic so threshold differences
#       are visible.

knowledge_base = [
    # Python / development
    # TODO: add 4–5 sentences
    "Python is a popular, high-level programming language known for its simplicity and readability.",
    " It is widely used in web development, data analysis, artificial intelligence, machine learning, and automation.",
     "Python has a large collection of libraries and frameworks that make programming easier and more efficient.",
     "It is beginner-friendly, making it a great choice for students who are learning to code.",
     "Its versatility and powerful features make Python one of the most widely used programming languages in the world.",

    # Another topic (e.g. cooking)
    # TODO: add 4–5 sentences
    "Cooking is an important life skill that allows people to prepare delicious and healthy meals.",
    "It involves using different ingredients, techniques, and tools to create a variety of dishes.",
    "Cooking can be a fun and creative activity that brings family and friends together.",
    "It also helps people learn about nutrition and develop healthy eating habits.",
    "Whether preparing a simple meal or trying a new recipe, cooking can be an enjoyable and rewarding experience.",
    # Another topic (e.g. space)
    # TODO: add 4–5 sentences
    "Space is a vast and mysterious place filled with stars, planets, galaxies, and other celestial objects.",
    "It is fascinating to explore because there are still many things scientists have not discovered.",
    "Astronauts travel into space to conduct experiments and learn more about the universe.",
    "The Moon, Mars, and other planets are important targets for future space exploration.",
    "Studying space helps us understand how the universe works and our place in it.",

    #4th topic
    "Pokémon is a popular franchise that features magical creatures with unique abilities and powers.",
    "Trainers travel around the world to catch, train, and battle Pokémon.",
    "Some of the most famous Pokémon include Pikachu, Charizard, and Bulbasaur.",
    "Pokémon games, movies, and TV shows have entertained fans of all ages for many years.",
    "The franchise teaches lessons about friendship, teamwork, and never giving up."
]
    

# ── Test queries ─────────────────────────────────────────────────────────────
# TODO: Create `test_queries` — at least 5 queries.
#       Include some that clearly match one topic and some that are ambiguous.

test_queries = [
    # TODO: add your queries here
    # Clearly related to one topic
    "How can I learn programming?",
    "How do I prepare healthy meals?",
    "What can astronauts discover in space?",
    "Which creatures can I catch and train?",

    # Ambiguous queries
    "What activities bring people together?",
    "What are some interesting things to explore?"
]

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model and embedding knowledge base...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
# TODO: Encode knowledge_base into `doc_embeddings`

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings=model.encode(knowledge_base)

# ── Run experiment ───────────────────────────────────────────────────────────
for query in test_queries:
    print(f'\nQuery: "{query}"')

    # TODO: Encode the query
    # TODO: Compute cosine similarity scores (1-D array)
    query_embeddings=model.encode([query])
    scores=cosine_similarity(query_embeddings,doc_embeddings)[0]

    for threshold in THRESHOLDS:

        # Filter sentences based on similarity score
        passing = [
            (score, sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score >= threshold
        ]

        # Sort from highest score to lowest
        passing.sort(reverse=True)

        # Print threshold and number of results
        print(f"  Threshold {threshold}: {len(passing)} results")

        # Print each matching sentence
        for score, sentence in passing:
            print(f"    [{score:.2f}] {sentence}")

        missed = [
            (score, sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score < threshold
        ]

        print("    Closest missed sentences:")

        for score, sentence in sorted(missed, reverse=True)[:2]:
            print(f"    [{score:.2f}] {sentence}")