"""
L7 — Prompt Workshop  (STARTER)
=================================
Run with:
    python starter.py

Your goal: write bad and good prompts for 3 tasks, send both to a model (or
mock), and compare outputs side by side.

Required features:
    1. A mock_llm() function so the script works without an API key
    2. An optional real_llm() function if you have an OpenAI key
    3. Three task pairs — each with a "bad" prompt and a "good" prompt
    4. Side-by-side output for each pair
    5. A STUDY_ASSISTANT_SYSTEM_PROMPT string at the bottom

Good prompt techniques to use (at least 2 per task):
    - Role framing:    "You are an expert Python instructor..."
    - Specificity:     exact format, length, or structure required
    - Few-shot:        one or two examples of the desired output
    - Output format:   "Respond only with valid JSON. No prose."

Task 1 — Code explanation:
    Get the model to explain what st.session_state does in Streamlit.

Task 2 — Data formatting:
    Convert a natural language task list into a JSON array with
    title, priority, and status fields.

Task 3 — System prompt design:
    Write a system prompt for a Course Study Assistant (see bottom of file).
"""

import os

# ── Mock LLM ─────────────────────────────────────────────────────────────────
def mock_llm(prompt: str) -> str:
    """Return a canned reply so the script runs without an API key."""
    # This is intentionally simple — replace with real_llm() when you have a key.
    return f"[MOCK RESPONSE]\nPrompt length: {len(prompt)} chars\nFirst 80 chars of prompt: {prompt[:80]!r}"


# ── Real LLM (optional) ───────────────────────────────────────────────────────
def real_llm(prompt: str, system: str = "") -> str:
    """Call OpenAI with the given prompt. Requires OPENAI_API_KEY env var."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


# Choose which function to use throughout this exercise
llm = mock_llm   # swap to real_llm if you have a key


# ── Helper ────────────────────────────────────────────────────────────────────
def compare(task_name: str, bad_prompt: str, good_prompt: str):
    """Run both prompts and print results side by side."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"\n--- BAD PROMPT ---\n{bad_prompt}\n")
    print(f"BAD RESPONSE:\n{llm(bad_prompt)}")
    print(f"\n--- GOOD PROMPT ---\n{good_prompt}\n")
    print(f"GOOD RESPONSE:\n{llm(good_prompt)}")


# ── Task 1 — Code explanation ─────────────────────────────────────────────────
# TODO: Write a vague bad_prompt asking about st.session_state
bad_prompt_1 = "explain st.session_state"

# TODO: Write a good_prompt using role framing + specificity (or few-shot)
#       e.g. ask for a beginner-friendly explanation with a concrete example
good_prompt_1 = "You are an expert Streamlit and Python instructor.I am an AI Engineering student who knows basic techniques and structures. can you explain what st.session_state is and how it is most likely implemented. Use 1 real world analogy before technical terms. keep this under 100 words"

compare("Code Explanation — st.session_state", bad_prompt_1, good_prompt_1)


# ── Task 2 — Data formatting ──────────────────────────────────────────────────
TASK_LIST = """
finish the module 7 exercises
review chunking strategies notes
watch the ChromaDB guided example video
start the module project
"""

# TODO: Write a vague bad_prompt asking to convert TASK_LIST to JSON
bad_prompt_2 = "convert TASK_LIST to JSON"

# TODO: Write a good_prompt that specifies the exact JSON structure
#       (array of objects with title, priority, status) and asks for
#       only valid JSON — no prose
good_prompt_2 = """
I am an AI Engineering student.
Convert the following task list into a JSON array of objects.

Each object must contain exactly three fields:
- "title": the task description
- "priority": either "high", "medium", or "low"
- "status": either "not started", "in progress", or "completed"

Assign a reasonable priority based on the task description.
Set the status of every task to "not started".

Use all four tasks from the task list.
Respond only with valid JSON. Do not include explanations,
Markdown formatting, or any additional text.

Task list:
finish the module 7 exercises
review chunking strategies notes
watch the ChromaDB guided example video
start the module project"""


compare("Data Formatting — task list to JSON", bad_prompt_2, good_prompt_2)


# ── Task 3 — System prompt design ────────────────────────────────────────────
# TODO: Write a bad_prompt asking the model to "help with course questions"
bad_prompt_3 = "Help me with my question"

# TODO: Write a good system prompt stored in STUDY_ASSISTANT_SYSTEM_PROMPT below,
#       then write a good_prompt that includes sample context and a question
STUDY_ASSISTANT_SYSTEM_PROMPT = """
You are an expert Course Study Assistant who helps students
understand their course materials.

Follow these rules:

1. Answer questions ONLY using the provided course context.
2. If the answer is not available in the context, clearly
   state that you do not know based on the provided materials.
3. Never invent information or make unsupported assumptions.
4. Keep every answer under 150 words.
5. Cite the source document used for your answer.
6. Explain concepts clearly using student-friendly language.

Your goal is to help students understand their course material
accurately and effectively.
"""

good_prompt_3 = """Course context:

Source document: Module 7 - Vector Databases Notes

Vector databases store information as numerical embeddings.
These embeddings represent the meaning of text, images, or
other data. They allow applications to search for information
based on semantic similarity rather than exact keyword matches.

Question:
What is a vector database, and why is it useful for AI applications?

Please answer using only the provided course context."""

compare("System Prompt Design — Study Assistant", bad_prompt_3, good_prompt_3)

# Print the final system prompt
print(f"\n{'='*60}")
print("STUDY ASSISTANT SYSTEM PROMPT:")
print(STUDY_ASSISTANT_SYSTEM_PROMPT)