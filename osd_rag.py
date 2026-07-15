"""
RAG Book Assistant — Streamlit + Mistral + Pinecone + MongoDB Atlas
---------------------------------------------------------------------
What's new in this version:

1. VECTOR DATABASE: Chroma -> Pinecone
   Pinecone is a managed, cloud-hosted vector database. Unlike Chroma's
   local `persist_directory`, Pinecone's index lives in the cloud, so
   it survives app restarts/redeploys and works on stateless hosting
   (Streamlit Cloud, Render, etc.) without needing a persistent disk.

2. CHATGPT-STYLE CHAT HISTORY
   Instead of one flat, ever-growing conversation, chats are now split
   into SESSIONS (like ChatGPT's left sidebar):
     - "+ New Chat" starts a fresh conversation
     - Every past session is listed in the sidebar, titled after its
       first question, newest on top
     - Clicking a past session reloads that conversation's messages
   Storage in MongoDB now has two collections:
     - chat_sessions: {_id, user_id, title, created_at}
     - chat_messages: {_id, session_id, role, content, created_at}

Setup:
  pip install streamlit pymongo langchain langchain-community \
      langchain-mistralai langchain-pinecone pinecone \
      langchain-text-splitters python-dotenv pypdf

  .env additions:
    MISTRAL_API_KEY=...
    MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority
    MONGODB_DB=rag_book_assistant
    PINECONE_API_KEY=...
    PINECONE_INDEX=rag-book-assistant
"""

import os
import tempfile
import uuid
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚", layout="wide")

MISTRAL_EMBED_DIMENSION = 1024  # mistral-embed output size

# ======================================================
# MongoDB Atlas — sessions + messages
# ======================================================


@st.cache_resource
def get_mongo_db():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "rag_book_assistant")

    if not uri:
        st.error("MONGODB_URI is not set in your .env file.")
        st.stop()

    client = MongoClient(uri)
    db = client[db_name]
    db["chat_sessions"].create_index([("user_id", 1), ("created_at", -1)])
    db["chat_messages"].create_index([("session_id", 1), ("created_at", 1)])
    return db


def create_session(db, user_id: str, title: str) -> str:
    session_id = str(uuid.uuid4())
    db["chat_sessions"].insert_one(
        {
            "_id": session_id,
            "user_id": user_id,
            "title": title[:60],
            "created_at": datetime.now(timezone.utc),
        }
    )
    return session_id


def list_sessions(db, user_id: str):
    try:
        return list(
            db["chat_sessions"]
            .find({"user_id": user_id})
            .sort("created_at", -1)
        )
    except PyMongoError as e:
        st.warning(f"Could not load past chats: {e}")
        return []


def load_messages(db, session_id: str):
    try:
        docs = db["chat_messages"].find({"session_id": session_id}).sort("created_at", 1)
        return [{"role": d["role"], "content": d["content"]} for d in docs]
    except PyMongoError as e:
        st.warning(f"Could not load messages: {e}")
        return []


def save_message(db, session_id: str, role: str, content: str):
    try:
        db["chat_messages"].insert_one(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": datetime.now(timezone.utc),
            }
        )
    except PyMongoError as e:
        st.warning(f"Could not save message: {e}")


# ======================================================
# Pinecone — vector database for PDF chunks
# ======================================================


@st.cache_resource
def get_pinecone_index_name():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "rag-book-assistant")

    if not api_key:
        st.error("PINECONE_API_KEY is not set in your .env file.")
        st.stop()

    pc = Pinecone(api_key=api_key)
    existing = [i.name for i in pc.list_indexes()]

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=MISTRAL_EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return index_name


def pinecone_has_vectors(index_name: str) -> bool:
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    stats = pc.Index(index_name).describe_index_stats()
    return stats.get("total_vector_count", 0) > 0


# ======================================================
# Sidebar: user id + ChatGPT-style session list
# ======================================================

with st.sidebar:
    st.subheader("👤 User")
    user_id = st.text_input("Username", value=st.session_state.get("user_id", ""))
    if user_id:
        st.session_state.user_id = user_id

if "user_id" not in st.session_state or not st.session_state.user_id:
    st.title("📚 RAG Book Assistant")
    st.info("👈 Enter a username in the sidebar to start chatting.")
    st.stop()

mongo_db = get_mongo_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = None  # None = unsaved new chat
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.divider()
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Your chats")

    sessions = list_sessions(mongo_db, st.session_state.user_id)
    for s in sessions:
        label = s["title"] or "New Chat"
        is_active = s["_id"] == st.session_state.session_id
        if st.button(
            ("💬 " if not is_active else "▶️ ") + label,
            key=f"session_{s['_id']}",
            use_container_width=True,
        ):
            st.session_state.session_id = s["_id"]
            st.session_state.messages = load_messages(mongo_db, s["_id"])
            st.rerun()

# ======================================================
# PDF upload + Pinecone vector DB creation
# ======================================================

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)

            embeddings = MistralAIEmbeddings(model="mistral-embed")
            index_name = get_pinecone_index_name()

            PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name,
            )

        st.success("Vector database created in Pinecone!")

# ======================================================
# Load Pinecone index + set up retriever / LLM
# ======================================================

index_name = get_pinecone_index_name()

if pinecone_has_vectors(index_name):

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    # Render full history of the ACTIVE session
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Enter your question")

    if query:
        # Lazily create the session in MongoDB on first message,
        # titled after the first question (ChatGPT-style auto-title).
        if st.session_state.session_id is None:
            st.session_state.session_id = create_session(
                mongo_db, st.session_state.user_id, title=query
            )

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query,
                    }
                )

                response = llm.invoke(final_prompt)
                answer = response.content
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        save_message(mongo_db, st.session_state.session_id, "user", query)
        save_message(mongo_db, st.session_state.session_id, "assistant", answer)

        st.rerun()  # refresh sidebar so a brand-new session shows up in the list

else:
    st.info("Upload a PDF and click 'Create Vector Database' to get started.")