"""
L7 — Chunk and Compare  (STARTER)
===================================
Run with:
    python starter.py

Your goal: implement two chunking strategies, embed both sets of chunks, run
the same queries against each, and compare results.

Required features:
    1. A document of at least 500 words (paste your own notes or write a
       multi-paragraph summary of course concepts)
    2. Fixed-size chunking: 300-character chunks, 50-character overlap
    3. Paragraph-based chunking: split on double newlines (\n\n)
    4. Embed all chunks from both strategies
    5. Run 3 different queries against both sets
    6. For each query display the top 2 results from each strategy with scores
    7. A written analysis at the end (as print statements)

Key concepts:
    Fixed-size chunking with overlap:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap

    Paragraph-based chunking:
        chunks = [p.strip() for p in text.split("\\n\\n") if p.strip()]

    Searching a list of embedded chunks:
        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:n]
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Document ─────────────────────────────────────────────────────────────────
# TODO: Paste or write a 500–800 word document here.
#       It should cover multiple distinct topics so different chunks are
#       retrievable. You can use course notes from Modules 1–6.

DOCUMENT = """
Learning Objectives
By the end of this lesson, you will be able to:
Explain why documents need to be split into chunks before embedding
Describe what a context window is and why it limits input size
Compare fixed-size chunking, sentence-based chunking, and semantic chunking
Identify the tradeoff between chunk size and search precision
Imagine you’re trying to find one specific fact in a 300-page textbook. Would you rather search page by page, chapter by chapter, or paragraph by paragraph?
Chapter by chapter: You’ll find the right chapter, but you’ll still need to read the whole thing to find the specific fact. Too much content dilutes the search.
Paragraph by paragraph: You’ll find exactly the right paragraph - but if the answer spans two paragraphs, you’ll only get half of it. Too little context breaks the meaning.
Page by page: A reasonable middle ground. Each page has enough context to be useful, and the search is precise enough to be helpful.
This is the chunking problem in AI systems. Before you can embed and search documents, you need to break them into pieces. How you break them matters enormously.
Why Not Embed Entire Documents?
Two reasons:
Context windows. Embedding models have a maximum input length. The all-MiniLM-L6-v2 model you’ve been using has a context window of 256 tokens (roughly 200 words). Any text beyond that is silently truncated - the model just ignores it. A 10-page document would lose 95% of its content.
Precision. Even if the model could handle the entire document, the resulting embedding would be an average of the entire document’s meaning. If a 50-page manual covers 20 topics, its embedding is a blurry average of all 20. When someone searches for one specific topic, the document’s embedding won’t be particularly close to the query - it’s spread too thin across too many concepts.
Chunking solves both problems: each chunk fits within the context window, and each chunk represents a focused piece of meaning.
What Is a Context Window?
A context window is the maximum number of tokens a model can process at once. A token is roughly 3/4 of a word ("embedding" = 1 token, "un-embed-ding" might be 3 tokens).
Different models have different context windows:
all-MiniLM-L6-v2: 256 tokens (~200 words)
all-mpnet-base-v2: 384 tokens (~300 words)
OpenAI text-embedding-3-small: 8,191 tokens (~6,000 words)
You need to choose chunks that fit within your embedding model’s context window. If a chunk exceeds the window, it gets truncated, and you lose information.
Chunking Strategies
1. Fixed-Size Chunking
Split text every N characters (or N words/tokens), with optional overlap:
def fixed_size_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap ensures we don't cut mid-idea
    return chunks

Pros: Simple, predictable chunk sizes, easy to implement.
Cons: Cuts mid-sentence, mid-paragraph, or mid-thought. A sentence about "Python FastAPI" might be split across two chunks.
2. Sentence-Based Chunking
Split on sentence boundaries, grouping sentences until reaching a target size:
import re

def sentence_chunks(text, max_chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\\s+', text)  # Split on sentence endings
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

Pros: Never cuts mid-sentence, preserves grammatical meaning.
Cons: Chunk sizes vary. Might still cut between related sentences.
3. Semantic/Paragraph Chunking
Split on natural document boundaries - paragraphs, sections, or headings:
Pros: Each chunk represents a complete thought or topic. Best for structured documents (documentation, articles, textbooks).
Cons: Paragraph sizes vary wildly. A 3-sentence paragraph and a 30-sentence paragraph produce very different chunk sizes.
The Chunk Size Tradeoff
Gemini_Generated_Image_y3geq7y3geq7y3ge.png

Small chunks (50–100 words): Very precise search - you find the exact sentence. But each chunk lacks context. "Use st.session_state" is less useful without the surrounding explanation of why and how.
Large chunks (500–1000 words): Rich context - each chunk tells a complete story. But the embedding becomes a blend of multiple ideas, reducing search precision.
The sweet spot is typically 200–500 words (or 150–400 tokens) for most applications. But there’s no universal "best" size - it depends on your documents and use case. In the module project, you’ll experiment with different sizes and compare the results.
Overlap: Why Chunks Should Slightly Overlap
If you cut a document at position 500, the idea that spans positions 490–510 gets split across two chunks. Neither chunk has the full thought.
Overlap solves this by having each chunk share some text with the next one. A 50-character overlap means positions 450–500 appear in both chunk N and chunk N+1. This ensures that ideas at chunk boundaries aren’t lost.
Typical overlap is 10–20% of chunk size (50–100 characters for a 500-character chunk).
Choosing a Strategy
Here’s a practical guide:
Structured documentation (like this course’s lessons): Paragraph/section chunking. The document already has natural breaks.
Conversational text (chat logs, support tickets): Sentence-based chunking.
Unstructured text (scraped web pages, PDFs): Fixed-size with overlap.
When in doubt: Start with fixed-size (300–500 characters, 50–100 overlap) and experiment from there.
In the next two lessons, you’ll store chunks in ChromaDB and build a search tool. The chunking strategy you choose directly affects the quality of your search results.
"""

# ── Chunking functions ───────────────────────────────────────────────────────
def fixed_size_chunks(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    # TODO: implement the sliding-window approach described above
    chunks = []
    start=0
    while start < len(text):
        end=min(start+size,len(text))
        chunks.append(text[start:end].strip())
        start+=size-overlap
    return chunks


def paragraph_chunks(text: str) -> list[str]:
    """Split text on double newlines, discarding empty paragraphs."""
    paragraphs = text.strip().split('\n\n')  # Split on double newlines
    return [p.strip() for p in paragraphs if p.strip()]


# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

fixed_chunks = fixed_size_chunks(DOCUMENT)
para_chunks  = paragraph_chunks(DOCUMENT)

print(f"Fixed-size chunks : {len(fixed_chunks)}")
print(f"Paragraph chunks  : {len(para_chunks)}")

# TODO: Encode fixed_chunks  → fixed_embeddings
# TODO: Encode para_chunks   → para_embeddings
fixed_embeddings=model.encode(fixed_chunks)
para_embeddings=model.encode(para_chunks)


# ── Search helper ────────────────────────────────────────────────────────────
def search_chunks(query: str, chunks: list[str], embeddings, n: int = 2):
    """Return top-n (score, chunk) pairs for a query."""

    # 1. Convert the query into an embedding
    query_emb = model.encode([query])

    # 2. Calculate cosine similarity
    scores = cosine_similarity(query_emb, embeddings)[0]

    # 3. Find the indices of the highest scores
    top_idx = np.argsort(scores)[::-1][:n]

    # 4. Return the scores and their corresponding chunks
    return [(float(scores[i]), chunks[i]) for i in top_idx]


# ── Test queries ─────────────────────────────────────────────────────────────
# TODO: Define 3 queries that target different parts of your document

queries = [
    "What is a chunk","what is a context window","what is the purpose of this lesson"
]

# ── Compare results ───────────────────────────────────────────────────────────
for query in queries:
    print(f'\n{"="*60}')
    print(f'Query: "{query}"')

    print("\n  [Fixed-size chunks]")
    fixed_results = search_chunks(
        query, fixed_chunks, fixed_embeddings, n=2
    )

    for score, chunk in fixed_results:
        print(f"\n    Score: {score:.4f}")
        print(f"    Chunk: {chunk[:300]}")

    print("\n  [Paragraph chunks]")
    para_results = search_chunks(
        query, para_chunks, para_embeddings, n=2
    )

    for score, chunk in para_results:
        print(f"\n    Score: {score:.4f}")
        print(f"    Chunk: {chunk[:300]}")

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ANALYSIS")
# TODO: Print your observations.
#       Which strategy returned more focused results?
#       Which missed context by cutting mid-sentence?
#       When would you choose each strategy?
print("""
1. Which strategy returned more focused results?

Paragraph-based chunking returned more focused results because it
keeps related sentences and ideas together. This makes it easier
for the embedding model to understand the meaning of each chunk
and find relevant information.

2. Which strategy missed context by cutting mid-sentence?

Fixed-size chunking sometimes missed context because it splits
the document every 300 characters. This can cut sentences or
separate related ideas between chunks. The 50-character overlap
helps reduce this problem but does not completely eliminate it.

3. When would you choose each strategy?

I would choose fixed-size chunking for large documents, scraped
web pages, or unstructured text because it creates consistent
chunk sizes and is easy to implement.

I would choose paragraph-based chunking for textbooks, articles,
and course notes because these documents already have natural
paragraph breaks that keep related information together.

4. Overall conclusion:

Both chunking strategies are useful, but the best choice depends
on the type of document being searched. Paragraph-based chunking
works well when preserving meaning and context is important,
while fixed-size chunking works well when consistent chunk sizes
are needed.
""")