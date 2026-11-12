"""
L7 — Expanded Semantic Search App  (STARTER)
=============================================
Run with:
    streamlit run starter.py

Your goal: extend a ChromaDB-backed Streamlit search tool with source
filtering, result counts, and expandable full text.

Required features:
    1. Load and chunk .txt / .md files from a docs/ folder
    2. Embed and store chunks in a ChromaDB collection (persistent)
    3. Sidebar multiselect to filter by source filename
    4. Search bar that queries ChromaDB (with optional where filter)
    5. "Showing X of Y total documents" count above results
    6. Each result: title (filename), preview (first 150 chars),
       st.expander showing full chunk text

Bonus (pick at least one):
    - A "Similar to this" button that re-queries using that chunk's text
    - Relevance badge (green/yellow/red) based on distance thresholds
    - Sidebar stat: total unique source files indexed

Key concepts:
    ChromaDB where filter for source files:
        where={"source": {"$in": selected_sources}}   # filter to selection
        # omit `where` entirely when no filter is active

    Reading all files from a folder:
        from pathlib import Path
        docs_dir = Path("docs")
        for file in docs_dir.glob("*.md"):
            text = file.read_text()
            # chunk and add to collection with metadata={"source": file.name}

    Counting total docs in collection:
        collection.count()
"""

import streamlit as st
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DOCS_DIR   = Path("docs")
CHUNK_SIZE = 500
OVERLAP    = 100
N_RESULTS  = 5

# ── Page config ──────────────────────────────────────────────────────────────
# TODO: st.set_page_config(page_title, page_icon, layout="wide")
st.set_page_config(page_title="Semantic Search App", page_icon="🔎", layout="wide")

# ── Model & ChromaDB ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    # TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # TODO: Create a PersistentClient with path="./chroma_db"
    client = chromadb.PersistentClient(path="./chroma_db")
    # TODO: Get or create collection "docs_search"
    # TODO: Return model and collection
    collection = client.get_or_create_collection(
        name="docs_search", configuration={"hnsw": {"space": "cosine"}})
    return model, collection

model, collection = load_resources()

# ── Index documents ───────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunking with overlap."""
    chunks = []

    step = size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        chunks.append(chunk)

    return chunks


    # ── Index documents ───────────────────────────────────────────────────────────
def index_docs():
    """Read docs/ folder and upsert all chunks into ChromaDB."""

    # Find all .txt and .md files
    files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.md"))

    for file in files:

        # Read document
        text = file.read_text(encoding="utf-8")

        # Split document into chunks
        chunks = chunk_text(text)

        if not chunks:
            continue

        # Generate embeddings
        embeddings = model.encode(chunks).tolist()

        # Create unique IDs
        ids = [f"{file.name}_chunk_{i}" for i in range(len(chunks))]

        # Create metadata
        metadatas = [{"source": file.name} for _ in chunks]

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────

# TODO: Get the list of all unique source filenames from the collection
#       Hint: collection.get(include=["metadatas"])["metadatas"]
# TODO: st.sidebar.multiselect to let users pick which sources to search
#       Store selection in `selected_sources`

st.sidebar.header("Filters")

with st.sidebar:
    st.title("📁 Document Manager")

    # Get unique source filenames from ChromaDB
    all_metadata = collection.get(include=["metadatas"])["metadatas"]

    sources = sorted(set(
        metadata["source"] for metadata in all_metadata
    ))

    # Source filename multiselect
    selected_sources = st.multiselect(
        "Filter by source",
        options=sources,
        default=sources
    )

    # Re-index documents
    if st.button("🔄 Re-index Documents"):
        index_docs()
        st.success("Documents re-indexed successfully!")

    # Display total documents in database
    st.metric("Documents in DB", collection.count())

    st.divider()

    # Number of search results
    n_results = st.slider(
        "Results to show",
        1,
        10,
        5
    )

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🔎 Semantic Search")

# Search input
query = st.text_input(
    "Search your documents",
    placeholder="Type a question or phrase..."
)

if query:

    # Build the source filter
    where = None

    if selected_sources and len(selected_sources) < len(sources):
        where = {"source": {"$in": selected_sources}}

    # Query ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Get total number of documents
    total = collection.count()

    # Get number of matching results
    num_results = len(results["documents"][0])

    st.write(f"Showing {num_results} of {total} total documents")

    st.divider()

    # Display results
    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Relevance badge
        if distance < 0.3:
            st.success("🟢 Highly Relevant")
        elif distance < 0.6:
            st.warning("🟡 Moderately Relevant")
        else:
            st.error("🔴 Less Relevant")

        # Source filename
        source = metadata.get("source", "Unknown")

        # Result heading
        st.subheader(f"📄 {source} — Distance: {distance:.4f}")

        # Preview (first 150 characters)
        preview = document[:150]

        if len(document) > 150:
            preview += "..."

        st.write(preview)

        # Expandable full text
        with st.expander("Show full text"):
            st.write(document)

        st.divider()

    # Handle empty results
    if num_results == 0:
        st.info("No matching documents found. Try another search.")
