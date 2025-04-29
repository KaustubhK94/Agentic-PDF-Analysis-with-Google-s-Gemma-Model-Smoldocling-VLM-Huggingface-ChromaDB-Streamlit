# agentic_rag_app.py
from pathlib import Path
import logging
import re
import requests
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.llms import Ollama
import ollama
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama.llms import OllamaLLM
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from agentic_workflow import run_end_to_end_pipeline
from utils import summarize_document, process_markdown_qa, save_summary_pdf
from config import PDF_PATH, IMAGES_FOLDER, CROPPED_FOLDER, COMBINED_MD, SUMMARY_MD, SUMMARY_PDF, CHROMA_DB_DIR
from langchain_core.tools import tool

# Configure structured logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s [query_id=%(query_id)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------ Constants ------------------
document_path = COMBINED_MD
llm = Ollama(model="gemma3:4b", base_url="http://localhost:11434")
ollama_model = "gemma3:4b"
llm_summarizer = OllamaLLM(model=ollama_model)

# ------------------ Build Vector Store ------------------
# Use cosine metric to ensure similarity scores in [0,1]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.80)

# Collection metadata for cosine distance
collection_metadata = {"hnsw:space": "cosine"}


def build_agent(markdown_path: str = COMBINED_MD):
    # Load and split markdown
    docs = UnstructuredMarkdownLoader(markdown_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

    # Create and persist Chroma vector DB with cosine metric
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_metadata=collection_metadata
    )
    vector_db.persist()

    # Wrap retriever with compression
    retriever = ContextualCompressionRetriever(
        base_retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        base_compressor=embeddings_filter
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Router prompt template to avoid direct str.format injection
    from string import Template
    router_template = Template(
        """
        Example 1:
        Question: What year was WaveNet introduced?
        Passages: "WaveNet was first published in 2016…"
        Answer: Yes

        Example 2:
        Question: What hyperparameter settings did they use?
        Passages: "We trained for 50 epochs on LJ Speech."
        Answer: No

        Now decide for this:
        User Question: $query
        Passages:
        $context

        Answer exactly "Answer: Yes" or "Answer: No".
        """
    )

    def check_local_knowledge(query: str, context: str) -> bool:
        prompt = router_template.safe_substitute(query=query, context=context)
        resp = llm.invoke(prompt)
        return resp.strip().lower() == "answer: yes"

    # Core QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    def qa_tool_fn(query: str) -> str:
        # 1. Retrieve docs with scores
        results = vector_db.similarity_search_with_score(query, k=5)
        if not results:
            logger.warning("No documents found for query, falling back to full-document QA", extra={"query_id": query[:8]})
            return process_markdown_qa(document_path, query)
        docs, distances = zip(*results)
        # Convert distance to similarity since using cosine metric
        sims = [1.0 - d for d in distances]
        max_sim = max(sims)
        logger.info(f"Max similarity: {max_sim}", extra={"query_id": query[:8]})
        # 2. Decide chain based on threshold and coverage
        context = "\n\n".join(d.page_content for d in docs)
        if max_sim < 0.75:
            logger.info("Similarity below threshold, full-document QA", extra={"query_id": query[:8]})
            return process_markdown_qa(document_path, query)
        if check_local_knowledge(query, context):
            logger.info("Using RetrievalQA chain", extra={"query_id": query[:8]})
            return qa_chain.run(query)
        else:
            logger.info("Router said no, full-document QA", extra={"query_id": query[:8]})
            return process_markdown_qa(document_path, query)
    
    @tool("summarize_tool", return_direct=True)
    def summarize_tool_fn(text: str) -> str:
        """
        Summarize the combined markdown document, and ask the user if they'd like to
        save it as a PDF.  The question will be returned as part of the assistant output,
        so it will show up in the Streamlit chat history.
        """
        summary = summarize_document(document_path, SUMMARY_MD)
        return summary 


    # Define tools
    qa_tool = Tool.from_function(
        qa_tool_fn,
        name="qa_tool",
        description="Detailed Q&A: retrieves, filters, and optionally falls back to full-document QA.",
        return_direct = True
    )

    summarize_tool = summarize_tool_fn
    
    # Initialize agent
    agent = initialize_agent(
        [qa_tool, summarize_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent

# ------------------ Main Loop (CLI) ------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting PDF extraction and agent build…", extra={"query_id": "startup"})
    run_end_to_end_pipeline(
        pdf_path=PDF_PATH,
        images_folder=IMAGES_FOLDER,
        cropped_folder=CROPPED_FOLDER,
        combined_md_filename=COMBINED_MD
    )
    logger.info("Extraction complete. Building agent index…", extra={"query_id": "startup"})
    agent = build_agent("COMBINED_MD")
    logger.info("Agent ready!", extra={"query_id": "startup"})
    while True:
        user_input = input("Q> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "escape"):
            print("👋 Goodbye!")
            break
        try:
            answer = agent.run(user_input)
        except Exception as e:
            answer = f"[Error during QA]: {e}"
            logger.error(answer, extra={"query_id": user_input[:8]})
        print(answer)
