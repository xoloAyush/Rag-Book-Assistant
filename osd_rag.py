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

from user_auth import require_auth, render_user_sidebar
from rate_limiter import (
    check_rate_limit,
    record_rate_limit_event,
    get_rate_limit_status,
)

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚", layout="wide")

MISTRAL_EMBED_DIMENSION = 1024  # mistral-embed output size

# Rate Limit Thresholds (Protected Routes)
RATE_LIMIT_UPLOAD_MAX = 3
RATE_LIMIT_UPLOAD_WINDOW = 3600    # 3 PDF uploads per hour

RATE_LIMIT_QUESTION_MAX = 10
RATE_LIMIT_QUESTION_WINDOW = 3600  # 10 questions per hour


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


def delete_session(db, session_id: str) -> bool:
    """Delete a chat session and all of its associated messages."""
    try:
        db["chat_sessions"].delete_one({"_id": session_id})
        db["chat_messages"].delete_many({"session_id": session_id})
        return True
    except PyMongoError as e:
        st.warning(f"Could not delete chat session: {e}")
        return False



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
# User Authentication & Session Setup
# ======================================================

mongo_db = get_mongo_db()
current_user = require_auth(mongo_db)
if current_user:
    render_user_sidebar(mongo_db, current_user)




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

    # Live Hourly Quota Display
    q_status = get_rate_limit_status(
        mongo_db, "ask_question", st.session_state.user_id, RATE_LIMIT_QUESTION_MAX, RATE_LIMIT_QUESTION_WINDOW
    )
    u_status = get_rate_limit_status(
        mongo_db, "pdf_upload", st.session_state.user_id, RATE_LIMIT_UPLOAD_MAX, RATE_LIMIT_UPLOAD_WINDOW
    )
    st.caption(
        f"⚡ **Hourly Quotas**\n\n"
        f"• Questions: **{q_status['used']}/{q_status['max']}**\n\n"
        f"• PDF Uploads: **{u_status['used']}/{u_status['max']}**"
    )

    st.divider()
    st.caption("Your chats")


    sessions = list_sessions(mongo_db, st.session_state.user_id)
    if not sessions:
        st.caption("No past chats yet.")
    for s in sessions:
        label = s["title"] or "New Chat"
        is_active = s["_id"] == st.session_state.session_id
        col_chat, col_del = st.columns([0.82, 0.18])
        with col_chat:
            if st.button(
                ("💬 " if not is_active else "▶️ ") + label,
                key=f"session_{s['_id']}",
                use_container_width=True,
            ):
                st.session_state.session_id = s["_id"]
                st.session_state.messages = load_messages(mongo_db, s["_id"])
                st.rerun()
        with col_del:
            if st.button(
                "🗑️",
                key=f"del_{s['_id']}",
                help=f"Delete '{label}'",
                use_container_width=True,
            ):
                delete_session(mongo_db, s["_id"])
                if st.session_state.session_id == s["_id"]:
                    st.session_state.session_id = None
                    st.session_state.messages = []
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

    upload_stat = get_rate_limit_status(
        mongo_db, "pdf_upload", st.session_state.user_id, RATE_LIMIT_UPLOAD_MAX, RATE_LIMIT_UPLOAD_WINDOW
    )
    st.caption(f"📊 Upload quota: {upload_stat['remaining']}/{upload_stat['max']} remaining this hour")

    if st.button("Create Vector Database"):
        allowed, rem, wait, wait_str = check_rate_limit(
            mongo_db, "pdf_upload", st.session_state.user_id, RATE_LIMIT_UPLOAD_MAX, RATE_LIMIT_UPLOAD_WINDOW
        )
        if not allowed:
            st.error(f"⚠️ PDF upload rate limit reached (maximum {RATE_LIMIT_UPLOAD_MAX} per hour). Please wait {wait_str} before uploading again.")
        else:
            with st.spinner("Processing document..."):
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(docs)

                embeddings = MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=os.getenv("MISTRAL_API_KEY")
                )
                index_name = get_pinecone_index_name()

                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name=index_name,
                )

                record_rate_limit_event(mongo_db, "pdf_upload", st.session_state.user_id)

            st.success("Vector database created in Pinecone!")
            st.rerun()


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

    llm = ChatMistralAI(
    model="codestral-latest",
    temperature=0,
    api_key=os.getenv("MISTRAL_API_KEY")
)

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
        allowed, rem, wait, wait_str = check_rate_limit(
            mongo_db, "ask_question", st.session_state.user_id, RATE_LIMIT_QUESTION_MAX, RATE_LIMIT_QUESTION_WINDOW
        )
        if not allowed:
            st.error(f"⚠️ Rate limit reached: Maximum {RATE_LIMIT_QUESTION_MAX} questions per hour. Please wait {wait_str} before asking another question.")
        else:
            record_rate_limit_event(mongo_db, "ask_question", st.session_state.user_id)

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

            st.rerun()  # refresh sidebar so quota indicators and new session show up


else:
    st.info("Upload a PDF and click 'Create Vector Database' to get started.")

    # uv run streamlit run osd_rag.py