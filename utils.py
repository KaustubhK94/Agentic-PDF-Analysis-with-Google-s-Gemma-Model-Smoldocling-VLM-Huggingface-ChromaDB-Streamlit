# my_streamlit_app/utils.py

from pathlib import Path
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
# Summarization imports
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama.llms import OllamaLLM
import logging
import markdown2
import pdfkit
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

import textwrap


llm = Ollama(model="gemma3:4b", base_url="http://localhost:11434")

ollama_model = "gemma3:4b"
llm_summarizer = OllamaLLM(model=ollama_model)

def format_summary(summary: str) -> str:
    """
    Apply basic formatting to the summary to improve readability.
    This includes adding extra newlines around LaTeX display math, code blocks, 
    and formatting bullet point lists.
    """
    # Add extra newlines before and after LaTeX display math ($$...$$)
    summary = re.sub(r'(\$\$)', r'\n\1', summary)
    summary = re.sub(r'(\$\$)', r'\1\n', summary)
    
    # Add extra newlines before and after code fences (``` ... ```)
    summary = re.sub(r'(```)', r'\n\1', summary)
    summary = re.sub(r'(```)', r'\1\n', summary)
    
    # Ensure bullet points have a newline before them
    summary = re.sub(r'\n([*-]\s)', r'\n\n\1', summary)
    
    # Optionally, wrap long lines for better readability (80-character width)
    wrapper = textwrap.TextWrapper(width=80)
    summary = "\n".join(wrapper.fill(line) for line in summary.splitlines())
    
    return summary.strip()

# prompt_template = """You are a biomedical research summarization expert. Create a concise, formal summary of the 
# following PubMed article text delimited by triple backticks.
# Return your response covering each key section marked with '##' in the text.
# Ensure that you:
# 1. Focus on key technicalities.
# 2. Maintain and clearly explain any technical or statistical details (e.g., contents from Tables, mathematical notation, equations, formulas, and quantitative metrics).
# 3. Preserve important quantitative results including statistical significance, effect sizes, confidence intervals, and p-values.
# 4. Emphasize details about the study population, interventions, and outcomes if provided.
# 5. Avoid speculative language or conversational phrasing.
# 6. Use precise, academic language appropriate for a biomedical audience.
# 7. Avoid follow up questions.
# 8. Maintain Any medical terms that are being mentioned like Radiological or Clinical features.
 

# ```{text}```
# BULLET POINT SUMMARY:
# """

prompt_template = """You are a research paper summarization expert. Create a concise,
formal summary of the following text delimited by triple backticks.
You're Summaries will be reviewed by academicians. 
Do not fabricate numbers or facts.if unsure say not enough information.
Return your response covering each key section marked with '##' in the text.
Ensure that you:
1. Focus on key technicalities,explicitely use information from the provided text. 
2. Maintain and explain mathematical or chemical notation, equations,formulas 
3. Preserve important quantitative results
4. Avoid speculative questions or conversational phrases.
5. Use academic language while explaining the given text content
6. Use precise, academic language.
7. Preserve important quantitative results including statistical.
8. Maintain Any medical terms that are being mentioned like Radiological or Clinical features.

```{text}```
BULLET POINT SUMMARY:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
summarize_chain = load_summarize_chain(llm_summarizer, chain_type="stuff", prompt=prompt)

def summarize_document(markdown_file_path: str, output_summary_path: str) -> str:
    """
    Loads the markdown file, extracts only true '## ' sections,
    summarizes each section, formats and saves the final summary.
    """
    try:
        text = Path(markdown_file_path).read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to load '{markdown_file_path}': {e}")
        return ""

    # Extract sections that start with '## ' and capture until next header
    pattern = r"(## [^\n]+[\s\S]*?)(?=\n## |\Z)"
    sections = re.findall(pattern, text)
    logging.info(f"Found {len(sections)} sections starting with '## '")

    summaries = []
    for idx, sec in enumerate(sections):
        # skip pure headers with no content
        body = re.sub(r'^##\s+[^\n]+', '', sec, flags=re.MULTILINE).strip()
        if not body:
            logging.info(f"Skipping empty section at index {idx}")
            continue
        try:
            doc = Document(page_content=sec)
            out = summarize_chain.invoke([doc]).get("output_text", "").strip()
            summaries.append((idx, out))
        except Exception as e:
            logging.error(f"Error summarizing section {idx}: {e}")
            summaries.append((idx, f"[Error: {e}]") )
    
    # Assemble, format, and save
    summaries.sort(key=lambda x: x[0])
    combined = "\n\n".join(text for _, text in summaries)
    formatted = format_summary(combined)

    try:
        Path(output_summary_path).write_text(formatted, encoding="utf-8")
        logging.info(f"Saved summary to '{output_summary_path}'")
    except Exception as e:
        logging.error(f"Failed to save summary: {e}")

    print("Formatted Final Summary:")
    print(formatted)
    return formatted


def process_markdown_qa(file_path: str, question: str, max_tokens=2500):
    """
    Reads markdown content, prepares a prompt for Q&A on the document,
    and generates an answer via Ollama.
    """
    md_content = Path(file_path).read_text(encoding='utf-8')
    prompt = f"""Analyze this markdown document and answer the question precisely.

    Document:
    {md_content}

    Question: {question}

    Consider markdown structure like headers, lists, and code blocks while answering.
    Answer:"""
    try:
        response = requests.post(
            'http://localhost:11434/api/tokenize',
            json={'model': 'gemma3:4b', 'prompt': prompt}
        )
        try:
            token_count = len(response.json().get('tokens', []))
        except Exception as e:
            print(f"Token counting error: {str(e)}")
        print(f"Token usage: {token_count}/131072")
    except Exception as e:
        print(f"Token counting skipped: {str(e)}")
    response = ollama.generate(
        model='gemma3:4b',
        prompt=prompt,
        options={
            'num_ctx': 131072,
            'temperature': 0.2,
            'num_predict': max_tokens
        }
    )
    return response['response']


def save_summary_pdf(md_path: str, pdf_path: str):
    """Convert a markdown file to PDF."""
    md = Path(md_path).read_text(encoding="utf-8")
    html = markdown2.markdown(md)
    pdfkit.from_string(html, pdf_path)
    print(f"✅ PDF saved as {pdf_path}")