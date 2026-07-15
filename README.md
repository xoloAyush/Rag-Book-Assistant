# 📚 RAG Book Assistant

A Streamlit app that lets you upload a PDF book and ask questions about it using Retrieval-Augmented Generation (RAG). Answers are grounded strictly in the uploaded document, chat history persists across sessions like ChatGPT, and everything is backed by cloud infrastructure rather than local files.

## Features

- **Chat with your PDF** — upload any PDF and ask natural-language questions about its content
- **Grounded answers only** — the assistant answers strictly from retrieved context and says so explicitly when the answer isn't in the document
- **ChatGPT-style chat history** — every conversation is saved as a session; switch between past chats or start a new one from the sidebar
- **Cloud-native storage** — no local database files; everything persists across restarts and deployments

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Mistral (`mistral-small-2506`) via `langchain-mistralai` |
| Embeddings | Mistral (`mistral-embed`) |
| Vector database | Pinecone (cloud, serverless) |
| Chat history database | MongoDB Atlas (cloud) |
| Orchestration | LangChain |

## How It Works

1. **Upload & index**: Upload a PDF → it's split into chunks → each chunk is embedded and stored in a Pinecone index.
2. **Ask questions**: Your question is embedded and matched against the most relevant chunks in Pinecone (MMR retrieval).
3. **Answer generation**: The retrieved chunks are passed as context to the LLM, which answers only from that context.
4. **Persistence**: Every question/answer pair is saved to MongoDB under the active chat session, so history survives page refreshes and future logins.

## Setup

### 1. Install dependencies

```bash
uv pip install streamlit pymongo pinecone langchain-pinecone langchain langchain-community langchain-mistralai langchain-text-splitters python-dotenv pypdf
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=rag_book_assistant
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-book-assistant
```

- Get a Mistral API key at [console.mistral.ai](https://console.mistral.ai)
- Get a MongoDB Atlas connection string at [mongodb.com/atlas](https://www.mongodb.com/atlas) (free tier works)
- Get a Pinecone API key at [pinecone.io](https://www.pinecone.io) (free serverless tier works)

### 3. Run the app

```bash
streamlit run osd_rag.py
```

## Usage

1. Enter a username in the sidebar (used to keep your chat history separate from other users)
2. Upload a PDF and click **Create Vector Database**
3. Ask questions in the chat box at the bottom
4. Use **➕ New Chat** to start a fresh conversation, or click any past chat in the sidebar to reopen it

## Known Limitations

- The Pinecone index is currently shared across all uploaded books — if you upload multiple different PDFs, retrieval may mix chunks from both. Per-book namespaces would fix this.
- User identification is a plain username with no authentication — anyone who knows/guesses a username can view that user's chat history.