import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

load_dotenv()

vectordb_file_path = "faiss_index"
DATASET_PATH = "ghostscan_ghostbot_dataset.csv"

# llm/embedder are built lazily (inside get_qa_chain(), on first use) rather
# than here at import time. Building them at import time means a bad/missing
# GROQ_API_KEY kills `from helper import get_qa_chain` in app.py before Flask
# even starts, or fails outside any try/except later - either way you get a
# generic HTML error page instead of a JSON error your route can catch and
# report clearly.
_llm = None
_embedder = None


def _get_llm():
    global _llm
    if _llm is None:
        # Accept either casing so a mismatch between this file and your .env
        # doesn't silently return None - os.getenv() is case-sensitive on
        # Linux/Mac.
        groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file next to "
                "app.py, then restart the Flask server."
            )
        # openai/gpt-oss-20b is a currently active Groq model.
        # llama-3.1-8b-instant has been deprecated by Groq and will raise a
        # 404 NotFoundError on every request if used here.
        _llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
    return _llm


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def create_vector_db():
    # Load data from the GhostBot Q&A dataset
    loader = CSVLoader(
        file_path=DATASET_PATH,
        source_column="Question",
        content_columns=["Answer"],
        metadata_columns=["Question"],
        encoding="utf-8",
        autodetect_encoding=True,  # allow trying other encodings if UTF-8 fails
    )

    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=_get_embedder())

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    # Build the index on first run instead of assuming it already exists -
    # app.py previously imported create_vector_db() but never called it,
    # so a fresh checkout would crash here with a missing-folder error.
    if not os.path.isdir(vectordb_file_path):
        create_vector_db()

    # Load the vector database from the local folder
    vectordb = FAISS.load_local(
        vectordb_file_path, _get_embedder(), allow_dangerous_deserialization=True
    )

    # A score_threshold retriever was filtering out every result, even exact
    # matches - FAISS's default similarity-score scale doesn't reliably land
    # in the 0-1 range that score_threshold expects, so a real match can
    # still score below the cutoff and get silently dropped. Plain top-k
    # retrieval is more reliable here: always return the 3 closest matches,
    # and let the prompt's own "if not found in context" instruction handle
    # genuinely unrelated questions instead.
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Given the following context and a question, generate an answer based on this context only. In the answer try to provide as much text as possible from the "Answer" field in the source document context without making much changes. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer. No Preamble.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Built with langchain_core primitives directly (LCEL) instead of
    # langchain.chains.RetrievalQA, which depends on the base `langchain`
    # package and repeatedly failed to import in testing.
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | _get_llm()
        | StrOutputParser()
    )

    return chain
