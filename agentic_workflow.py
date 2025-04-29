import os
import re
import logging
import time
from pathlib import Path
from typing import List, Dict

import torch
from pdf2image import convert_from_path
from PIL import Image
import ollama  # Latest Ollama Python client should be installed
import textwrap
import requests


# Import Docling components
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from langchain_core.output_parsers import StrOutputParser

# LangChain imports for summarization and Q&A
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama.llms import OllamaLLM
from IPython import get_ipython
import sys
from PyPDF2 import PdfReader
from config import ROOT, COMBINED_MD, SUMMARY_MD


# Set device for torch processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Step 1: PDF to Images
# -------------------------------
def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 200) -> List[str]:
    """
    Convert a PDF into images and save each page to the specified folder.
    Returns a list of image file paths.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for idx, page in enumerate(pages, start=1):
        image_filename = f"page_{idx:03d}.png"
        image_path = os.path.join(output_folder, image_filename)
        page.save(image_path, "PNG")
        logging.info(f"Saved page {idx} as {image_path}")
        image_paths.append(image_path)
    return image_paths

# -------------------------------
# vLLM Setup for Fast Inference
# -------------------------------
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM: pip install vllm")

MODEL_PATH = "ds4sd/SmolDocling-256M-preview"  # Change if needed
PROMPT_TEXT = "Convert page to Docling."
CHAT_TEMPLATE = f"""<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>
Assistant:"""

# vllm_instance = LLM(model=MODEL_PATH, dtype="half", enforce_eager=True, limit_mm_per_prompt={"image": 1})
# sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
_vllm = None
_sampling = None

# -------------------------------
# Extract Image Regions from DocTags
# -------------------------------
def extract_image_locations_from_doctags(doctags: str, base_name: str) -> List[Dict]:
    """
    Extract image regions from the DocTag string.
    Returns a list of dictionaries containing:
      - tag: the type (picture, chart, etc.)
      - coords: tuple of (x1, y1, x2, y2)
      - caption: caption text if present
      - y_position: top coordinate for vertical ordering
      - filename: a generated filename for the cropped image
    """
    images = []
    image_tag_pattern = r"<(picture|chart|flow_chart)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>(.*?)</(picture|chart|flow_chart)>"
    for idx, match in enumerate(re.finditer(image_tag_pattern, doctags, re.DOTALL)):
        tag = match.group(1)
        x1, y1, x2, y2 = map(int, (match.group(2), match.group(3), match.group(4), match.group(5)))
        inner_content = match.group(6)
        caption_match = re.search(r"<caption>(.*?)</caption>", inner_content, re.DOTALL)
        caption = caption_match.group(1).strip() if caption_match else ""
        filename = f"{base_name}_{tag}_{idx+1:02d}.png"
        images.append({
            "tag": tag,
            "coords": (x1, y1, x2, y2),
            "caption": caption,
            "y_position": y1,
            "filename": filename
        })
    images.sort(key=lambda x: x["y_position"])
    return images

# -------------------------------
# Insert Image Markers in Markdown Based on Captions
# -------------------------------
def insert_image_markers(markdown: str, image_locations: List[Dict]) -> str:
    """
    Insert markers into the markdown at positions corresponding to image captions.
    For each image region, if its caption exists in the markdown text, insert a marker
    immediately following the first occurrence of the caption; otherwise, append at the end.
    """
    for img in image_locations:
        marker = f"<!-- IMG:{img['filename']} -->"
        if img["caption"]:
            caption_pattern = re.escape(img["caption"])
            match = re.search(caption_pattern, markdown)
            if match:
                pos = match.end()
                markdown = markdown[:pos] + "\n" + marker + "\n" + markdown[pos:]
            else:
                markdown += "\n" + marker + "\n"
        else:
            markdown += "\n" + marker + "\n"
    return markdown

# -------------------------------
# Process a Single Image (Individual Processing)
# -------------------------------

def get_vllm():
    global _vllm, _sampling
    if _vllm is None:
        MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
        _vllm = LLM(model=MODEL_PATH, dtype="half", enforce_eager=True, limit_mm_per_prompt={"image":1})
        _sampling = SamplingParams(temperature=0.0, max_tokens=16384)
    return _vllm, _sampling

def process_single_image(image_path: str, output_folder: str) -> None:
    """
    Process an individual image:
    - Run vLLM inference to obtain DocTags.
    - Save the DocTags file alongside the image.
    - Generate markdown from the DocTags using DoclingDocument.
    - Insert inline image markers based on extracted coordinates.
    - Save the markdown output to a file.
    """
    base_name = Path(image_path).stem
    try:
        vllm, sampling = get_vllm()
        image = Image.open(image_path).convert("RGB")
        llm_input = {"prompt": CHAT_TEMPLATE, "multi_modal_data": {"image": image}}
        output = vllm.generate([llm_input], sampling_params=sampling)[0]
        doctags = output.outputs[0].text
        logging.info(f"Processed {image_path} -> DocTags output length: {len(doctags)}")

        # Save the DocTags file
        doctags_filename = Path(image_path).with_suffix(".doctags")
        with open(doctags_filename, "w", encoding="utf-8") as f:
            f.write(doctags)
        logging.info(f"DocTags saved to {doctags_filename}")

        # Extract image regions with coordinate info and generate filenames
        image_locations = extract_image_locations_from_doctags(doctags, base_name)

        # Generate markdown via DoclingDocument export
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name=f"Document_from_{base_name}")
        doc.load_from_doctags(doctags_doc)
        markdown_output = doc.export_to_markdown()

        # Insert markers in the markdown at appropriate positions
        markdown_output = insert_image_markers(markdown_output, image_locations)

        # Save the final markdown output to a file named as the page image (e.g. page_001.md)
        md_filename = Path(image_path).with_suffix(".md")
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        logging.info(f"Markdown output saved to {md_filename}")
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        raise


def combine_markdowns(image_files: List[str], combined_md_filename: str):
    combined_content = ""
    for i, image_path in enumerate(image_files, start=1):
        md_file = Path(image_path).with_suffix(".md")
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                page_content = f.read()
                combined_content += f"!! Page {i}\n\n" + page_content + "\n\n"
    with open(combined_md_filename, "w", encoding="utf-8") as f:
        f.write(combined_content)
    logging.info(f"Combined markdown saved to {combined_md_filename}")


# -------------------------------
# Crop Pictorial Elements using DocTags Coordinates
# -------------------------------
def crop_and_save_elements(page_image_path: str, doctags_path: str, output_folder: str, scale_to_image: bool = True):
    """
    Crop and save image elements from a page image using coordinate data from the DocTags.
    Cropped images use a filename that matches the marker.
    """
    page_image = Image.open(page_image_path)
    width, height = page_image.size
    base_name = Path(page_image_path).stem

    with open(doctags_path, "r", encoding="utf-8") as f:
        doctags = f.read()

    image_regions = extract_image_locations_from_doctags(doctags, base_name)
    os.makedirs(output_folder, exist_ok=True)
    
    for region in image_regions:
        tag = region["tag"]
        x1, y1, x2, y2 = region["coords"]
        if scale_to_image:
            x1 = int(x1 / 500.0 * width)
            y1 = int(y1 / 500.0 * height)
            x2 = int(x2 / 500.0 * width)
            y2 = int(y2 / 500.0 * height)
        cropped = page_image.crop((x1, y1, x2, y2))
        out_filename = region["filename"]
        out_path = os.path.join(output_folder, out_filename)
        cropped.save(out_path)
        logging.info(f"Saved cropped {tag} to {out_path}")

# -------------------------------
# Describe Cropped Images via Gemma3:4b
# -------------------------------
def describe_image_with_gemma3(image_path: str, user_message: str = "Describe the image in detail.") -> str:
    try:
        response = ollama.chat(
            model="gemma3:4b",
            messages=[{
                "role": "user",
                "content": user_message,
                "images": [image_path]
            }]
        )
        
        # If response has a message attribute with content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content.strip()
            
        # If response is a dictionary with message
        if isinstance(response, dict) and "message" in response:
            if isinstance(response["message"], dict) and "content" in response["message"]:
                return response["message"]["content"].strip()
        
        # Fallback to regex extraction from string representation
        response_str = str(response)
        import re
        content_match = re.search(r"content=['\"](.*?)['\"](?=, images=|, tool_calls=|\)$)", response_str, re.DOTALL)
        if content_match:
            content = content_match.group(1)
            # Unescape any escaped quotes and backslashes
            content = content.replace("\\'", "'").replace("\\\\", "\\").replace("\\n", "\n")
            return content.strip()
        
        # Last resort - return the whole response
        return str(response)
        
    except Exception as e:
        logging.error(f"Error calling Ollama for image description: {e}")
        return f"[Error obtaining description: {e}]"

def process_cropped_images_and_get_descriptions(cropped_folder: str) -> Dict[str, str]:
    """
    Process all cropped images in the given folder and return a dictionary mapping filenames to descriptions.
    """
    desc_dict = {}
    for image_file in sorted(Path(cropped_folder).glob("*.png")):
        desc = describe_image_with_gemma3(str(image_file))
        desc_dict[image_file.name] = desc
        logging.info(f"Obtained description for {image_file.name}")
    return desc_dict

# -------------------------------
# LangChain Summarization Chain Setup (unchanged)
# -------------------------------
ollama_model = "gemma3:4b"
llm = OllamaLLM(model=ollama_model)

def format_summary(summary: str) -> str:
    summary = re.sub(r'<sub>(.*?)</sub>', r'_{\1}', summary)
    summary = re.sub(r'<sup>(.*?)</sup>', r'^{\1}', summary)
    summary = re.sub(r'(\$\$)', r'\n\1', summary)
    summary = re.sub(r'(\$\$)', r'\1\n', summary)
    summary = re.sub(r'(```)', r'\n\1', summary)
    summary = re.sub(r'(```)', r'\1\n', summary)
    summary = re.sub(r'\n([*-]\s)', r'\n\n\1', summary)
    wrapper = textwrap.TextWrapper(width=80)
    return "\n".join(wrapper.fill(line) for line in summary.splitlines()).strip()

# prompt_template = """You are a research paper summarization expert. Create a concise, formal summary of the following text delimited by triple backticks.
# Return your response covering each key section marked with '##' in the text.
# Ensure that you:
# 1. Focus on key technicalities 
# 2. Maintain and explain mathematical notation, equations,formulas 
# 3. Preserve important quantitative results
# 4. Avoid speculative questions or conversational phrases
# 5. Use academic language

# ```{text}```
# BULLET POINT SUMMARY:
# """

# prompt_template = """You are a biomedical research summarization expert. Create a concise, formal summary of the following PubMed article text delimited by triple backticks.
# Return your response covering each key section marked with '##' in the text.
# Ensure that you:
# 1. Focus on key technicalities.
# 2. Maintain and clearly explain any technical or statistical details (e.g., contents from Tables, mathematical notation, equations, formulas, and quantitative metrics).
# 3. Preserve important quantitative results including statistical significance, effect sizes, confidence intervals, and p-values.
# 4. Emphasize details about the study population, interventions, and outcomes if provided.
# 5. Avoid speculative language or conversational tones.
# 6. Use precise, academic language appropriate for a biomedical audience.
# 7. Avoid follow up questions.
# 8. Maintain Any medical terms that are being mentioned like Radiological or Clinical features 

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

# -------------------------------
# End-to-end Pipeline Integration (Individual Image Processing)
# -------------------------------
def run_end_to_end_pipeline(pdf_path: str, images_folder: str, cropped_folder: str, combined_md_filename: str):
    """
    Complete workflow:
      1. Convert PDF pages to images.
      2. For each image (page), process it individually:
         - Generate DocTags and markdown output.
         - Save the DocTags file and markdown output alongside each page image.
      3. Release GPU memory.
      4. Crop image regions from each page using the saved DocTags.
      5. Get image descriptions.
      6. Replace markers in each markdown file with the corresponding image descriptions.
    """
    # 0) Check page count
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        logging.error(f"Unable to read PDF pages: {e}")
        raise RuntimeError("Failed to open PDF for page‐count check.")
    
    if num_pages > 15:
        raise ValueError(f"PDF has {num_pages} pages, but maximum supported is 15. "
                         "Please upload a shorter document.")
    
    logging.info(f"PDF page count = {num_pages}; proceeding with pipeline.")
    
    # Convert PDF to images.
    image_files = pdf_to_images(pdf_path, images_folder)
    
    # Process each image individually.
    for image_path in image_files:
        process_single_image(image_path, output_folder=images_folder)
    
    # Release GPU memory.
    try:
        global _vllm, _sampling
        del  _vllm, _sampling
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cleared after markdown generation")
    except Exception as e:
        logging.error(f"Error clearing GPU memory: {e}")
    
    # Crop images from each page (using saved DocTags).
    for image_file in image_files:
        doctags_file = Path(image_file).with_suffix(".doctags")
        if doctags_file.exists():
            crop_and_save_elements(str(image_file), str(doctags_file), cropped_folder)
        else:
            logging.warning(f"DocTags file not found for {image_file}")
    
    # For each markdown file, replace markers with descriptions.
    desc_dict = process_cropped_images_and_get_descriptions(cropped_folder)
    
    # Save descriptions to JSON
    import json
    desc_file = Path(cropped_folder)/ "descriptions.json"
    with open(desc_file, "w") as f:
        json.dump(desc_dict, f)

    # Process each individual markdown file.
    for image_path in image_files:
        md_file = Path(image_path).with_suffix(".md")
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_text = f.read()
            def replace_marker(match):
                fname = match.group(1).strip()
                desc = desc_dict.get(fname, "[No description found]")
                img_path = f"{cropped_folder}/{fname}"
                return (
                    f"\n"
                    f"![{fname}]({img_path})\n\n"       # 1) Markdown image
                    f"**Image {fname}:** {desc}\n"    # 2) then its description
                    )
            final_md = re.sub(r"<!-- IMG:(.*?) -->", replace_marker, markdown_text)
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(final_md)
            logging.info(f"Final markdown with inline image descriptions saved to {md_file}")
    
    combine_markdowns(image_files, combined_md_filename)
    logging.info("End-to-end pipeline completed.")

# -------------------------------
# Agentic Workflow Integration
# -------------------------------
class AgenticAIAgent:
    def __init__(self, pdf_path: str, images_folder: str, cropped_folder: str, combined_md_filename: str, output_summary_path: str):
        self.pdf_path = pdf_path
        self.images_folder = images_folder
        self.cropped_folder = cropped_folder
        self.combined_md_filename = combined_md_filename
        self.output_summary_path = output_summary_path

    
    def run_pipeline(self):
        logging.info("Agent starting full pipeline run...")
        run_end_to_end_pipeline(self.pdf_path, self.images_folder, self.cropped_folder, self.combined_md_filename)
        logging.info("Pipeline run complete.")
    
    def ensure_pipeline(self):
        if not os.path.exists(self.combined_md_filename):
            print("Combined markdown not found. Running pipeline...")
            self.run_pipeline()
        return True
    
 
    def summarize_document(self):
        self.ensure_pipeline()
        # Pass self.combined_md_filename directly (it contains the file path)
        summary = summarize_document(self.combined_md_filename, output_summary_path=self.output_summary_path)
        print("Formatted Final Summary:")
        print(summary)
        summary_filename = Path(self.combined_md_filename).with_stem(Path(self.combined_md_filename).stem + "_summary").with_suffix(".md")
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to {summary_filename}")

    
    def qa_on_document(self, question: str):
        self.ensure_pipeline()
        answer = process_markdown_qa(self.combined_md_filename, question)
        print("Answer:")
        print(answer)


def process_markdown_qa(file_path: str, question: str, max_tokens=500):
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
        token_count = len(response.json()['tokens'])
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

# -------------------------------
# Main Entrypoint (Jupyter and CLI)
# -------------------------------
def main_jupyter(pdf_path: str = str(ROOT / "media" / "uploads" / "input.pdf"),
                 combined_md: str = str(COMBINED_MD),
                 output_summary: str = str(SUMMARY_MD)):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    agent = AgenticAIAgent(
        pdf_path=pdf_path,
        images_folder=str(ROOT / "media" / "pdf_images"),
        cropped_folder=str(ROOT / "media" / "crops"),
        combined_md_filename=combined_md,
        output_summary_path=output_summary
    )
    print("🔍 Running PDF extraction pipeline...")
    agent.run_pipeline()
    print("✅ Extraction complete!")

    while True:
        choice = input(
            "\nEnter 'summary' for document summary, "
            "'qa' for Q&A, or 'exit' to quit: "
        ).strip().lower()
        if choice == "exit":
            print("👋 Goodbye!")
            break
        elif choice == "summary":
            agent.summarize_document()
        elif choice == "qa":
            question = input("Enter your Q&A question: ").strip()
            agent.qa_on_document(question)
        else:
            print("❓ Invalid choice. Please type 'summary', 'qa', or 'exit'.")

def main():
    # If in Jupyter/IPython, hand off to the interactive loop
    try:
        get_ipython  # type: ignore
        print("👩‍💻 Detected Jupyter environment — launching interactive mode.")
        return main_jupyter()
    except NameError:
        pass

    import argparse
    parser = argparse.ArgumentParser(
        description="📄 Agentic PDF Analysis: extraction, RAG, summarization"
    )
    parser.add_argument(
        "--pdf", "-p",
        required=True,
        help="Path to input PDF file (max 15 pages)."
    )
    parser.add_argument(
        "--combined_md", "-c",
        default=str(COMBINED_MD),
        help="Where to write the intermediate combined markdown."
    )
    parser.add_argument(
        "--output_path", "-o",
        default=str(SUMMARY_MD),
        help="Where to write the final summary markdown."
    )
    args = parser.parse_args()

    # 0) Page‐count check
    try:
        reader = PdfReader(args.pdf)
        num_pages = len(reader.pages)
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}", file=sys.stderr)
        sys.exit(1)

    if num_pages > 15:
        print(
            f"❗ PDF has {num_pages} pages; exceeds 15 page limit. "
            "Please provide a shorter document.",
            file=sys.stderr
        )
        sys.exit(1)

    # 1) Run pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    agent = AgenticAIAgent(
        pdf_path=args.pdf,
        images_folder=str(ROOT / "media" / "pdf_images"),
        cropped_folder=str(ROOT / "media" / "crops"),
        combined_md_filename=args.combined_md,
        output_summary_path=args.output_path
    )

    print(f"✔ PDF OK ({num_pages} pages). Starting extraction…")
    try:
        agent.run_pipeline()
        print("✅ Extraction complete!")
    except Exception as e:
        print(f"❌ Pipeline error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Generate summary
    print(f"🔍 Generating summary to '{args.output_path}' …")
    try:
        agent.summarize_document()
        print("✅ Summary complete!")
    except Exception as e:
        print(f"❌ Summarization error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()