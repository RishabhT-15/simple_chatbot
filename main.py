import os
import shutil
import tempfile
import zipfile
import re
import chromadb
import torch
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import streamlit as st
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import sys
import importlib

importlib.import_module("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]


# -------------- Configuration --------------

# Groq API config - replace with your actual Groq API key or keep hardcoded as you prefer
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Secure in prod
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Session state for per-user repo embeddings
if "user_id" not in st.session_state:
    st.session_state["user_id"] = "user_" + next(tempfile._get_candidate_names())

BASE_PERSIST_DIR = "./chroma_db"
os.makedirs(BASE_PERSIST_DIR, exist_ok=True)

# Use CPU for sentence-transformers for Mac M1 compatibility
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)


# ------------ Helper functions ------------

def clean_collection_name(name: str) -> str:
    # Clean user_id to match ChromaDB collection naming rules
    name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    name = name[:512]
    if len(name) < 3:
        name = name.ljust(3, '0')
    return name


def collect_and_chunk(directory, extensions=('.py', '.java', '.js', '.md')):
    chunks, metadata = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception:
                    continue
                for chunk in text_splitter.split_text(text):
                    chunks.append(chunk)
                    metadata.append({'source': path})
    return chunks, metadata


def embed_uploaded_repo(zip_bytes, raw_user_id):
    user_id = clean_collection_name(raw_user_id)
    user_dir = os.path.join(tempfile.gettempdir(), user_id)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(user_dir, exist_ok=True)
    zip_path = os.path.join(user_dir, "repo.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(user_dir)
    except Exception as e:
        raise RuntimeError(f"Invalid or corrupted repo archive: {str(e)}")

    chunks, metadata = collect_and_chunk(user_dir)
    persist_dir = os.path.join(BASE_PERSIST_DIR, user_id)
    os.makedirs(persist_dir, exist_ok=True)

    chroma_client = PersistentClient(path=persist_dir)
    try:
        chroma_client.delete_collection(user_id)
    except Exception:
        pass

    collection = chroma_client.create_collection(user_id)
    if chunks:
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
            ids=[str(i) for i in range(len(chunks))]
        )
    else:
        raise RuntimeError("No code chunks found in the uploaded repository.")
    return len(chunks)


def retrieve_context(query, raw_user_id, top_k=3):
    user_id = clean_collection_name(raw_user_id)
    persist_dir = os.path.join(BASE_PERSIST_DIR, user_id)
    chroma_client = PersistentClient(path=persist_dir)
    try:
        collection = chroma_client.get_collection(user_id)
    except Exception as e:
        raise RuntimeError(f"Collection for user '{user_id}' not found. Please upload repo first. ({str(e)})")

    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results['documents'][0] if results['documents'] else []
    return docs


def query_groq_api(prompt, max_new_tokens=600):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # Prepare messages payload with system prompt + user prompt:
    system_msg = {
        "role": "system",
        "content": "You are a professional coding assistant for the Ghost Android post-exploitation framework."
    }
    user_msg = {
        "role": "user",
        "content": prompt
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [system_msg, user_msg],
        "max_tokens": max_new_tokens,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    # Debug info (optional)
    # print(f"Status code: {response.status_code}")
    # print(f"Response text: {response.text}")

    if response.status_code != 200:
        raise RuntimeError(f"Groq API request failed: {response.status_code} {response.text}")

    result = response.json()
    # Extract the generated content
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Groq API response structure: {result}")


def chat_with_rag(query, user_id):
    context_docs = retrieve_context(query, user_id)
    context = "\n".join(context_docs)
    prompt = (
        f"Below is a snippet of the Ghost codebase for context:\n"
        f"{context}\n\n"
        f"{query}\n"
        "Provide only the complete Python module code, wrapped exactly as shown:\n"
        "### START OF YOUR CODE ###\n"
        "<code here>\n"
        "### END OF YOUR CODE ###"
    )
    output_text = query_groq_api(prompt, max_new_tokens=600)

    start_marker = "### START OF YOUR CODE ###"
    end_marker = "### END OF YOUR CODE ###"
    start_idx = output_text.find(start_marker)
    end_idx = output_text.find(end_marker, start_idx)

    if start_idx != -1 and end_idx != -1:
        code = output_text[start_idx + len(start_marker):end_idx].strip()
    else:
        code = output_text.strip()
    return code


# --------------- Streamlit UI -------------------

st.set_page_config(page_title="RAG Chatbot (Repo-Aware) - Groq API", layout="wide")
st.markdown(
    "<h2 style='margin-bottom:1rem'>RAG Chatbot with Repo Upload (Using Groq API)</h2>",
    unsafe_allow_html=True,
)

# Sidebar: Repo Upload
st.sidebar.header("Convert Your Repo")
repo_file = st.sidebar.file_uploader(
    "Drop a zipped repo here (Python/Java/JS/MD files supported):",
    type=["zip"]
)

if repo_file:
    st.sidebar.info("Processing repo...")
    try:
        num_chunks = embed_uploaded_repo(repo_file.read(), st.session_state["user_id"])
        st.sidebar.success(f"Repo indexed! {num_chunks} chunks.")
    except Exception as e:
        st.sidebar.error(f"Error processing repo: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Use the chat below to ask code/context questions. The model will use your uploaded repo for context!")

# Main chat area
st.markdown("#### Chat with LLM (using repo context if uploaded)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

chat_input = st.text_area("Your message:", key="chat_in", height=100)

if st.button("Send", key="send_btn") and chat_input.strip():
    st.session_state["messages"].append(("User", chat_input))
    with st.spinner("Generating answer..."):
        try:
            result = chat_with_rag(chat_input, st.session_state["user_id"])
            st.session_state["messages"].append(("Bot", result))
        except Exception as e:
            st.session_state["messages"].append(("Bot", f"Error: {e}"))
    st.rerun()



#chat history

for role, msg in st.session_state.get("messages", []):
    color = "#2e4737" if role == "Bot" else "#24325a"

    # Split message into alternating [text, code, text, code ...] components
    # Matches code blocks of form ``````
    pattern = re.compile(r"``````", re.DOTALL)
    last_end = 0
    blocks = []

    for m in pattern.finditer(msg):
        if m.start() > last_end:
            # Text before code block
            blocks.append(('text', msg[last_end:m.start()]))
        language = m.group(1)
        code = m.group(2)
        blocks.append(('code', code, language))
        last_end = m.end()
    if last_end < len(msg):
        blocks.append(('text', msg[last_end:]))

    # Print the role label only at the top of the first block
    role_label = f"<b>{role}:</b><br>"

    for i, block in enumerate(blocks):
        if block[0] == 'text':
            content = block[1].strip()
            if content:
                st.markdown(
                    f"""
                    <div style="
                        margin:1em 0;
                        padding:1em;
                        border-radius:8px;
                        background:{color};
                        color:#f2f2f2;
                        overflow-x:auto;
                        font-size: 1.04em;">
                        {role_label if i==0 else ""}{content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif block[0] == 'code':
            code, lang = block[1], block[2] or None
            st.code(code, language=lang)
        else:
            pass  # Should not happen