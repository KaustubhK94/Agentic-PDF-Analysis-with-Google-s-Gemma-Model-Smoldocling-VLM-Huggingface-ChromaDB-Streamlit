# my_streamlit_app/app_new.py
import streamlit as st
from pathlib import Path
import json
import base64
import markdown2
import sys
from PyPDF2 import PdfReader

from agentic_workflow import run_end_to_end_pipeline
from config import (
    PDF_PATH, IMAGES_FOLDER, CROPPED_FOLDER,
    COMBINED_MD, SUMMARY_MD, SUMMARY_PDF
)
from utils import save_summary_pdf
from agentic_rag_app import build_agent

st.set_page_config(layout="wide")


# ensure dirs exist
for d in ["my_streamlit_app/media/uploads", IMAGES_FOLDER, CROPPED_FOLDER]:
    Path(d).mkdir(parents=True, exist_ok=True)
    
# --- SIDEBAR: Upload + Extraction + Scrollable PDF Preview ---
with st.sidebar:
    st.header("📄 PDF Upload & Preview")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        save_path = Path("my_streamlit_app/media/uploads") / uploaded_pdf.name
        save_path.write_bytes(uploaded_pdf.getbuffer())
        if st.button("Run Extraction"):
            try:
                with open(save_path, "rb") as f:
                    reader = PdfReader(f)
                    if len(reader.pages) > 15:
                        st.error(f"❗ This PDF is {len(reader.pages)} pages long. "
                             "Please upload a document with 15 pages or fewer.")
                        st.stop()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                st.stop()

            with st.spinner("Extracting PDF…"):
                run_end_to_end_pipeline(
                    pdf_path=str(save_path),
                    images_folder=IMAGES_FOLDER,
                    cropped_folder=CROPPED_FOLDER,
                    combined_md_filename=COMBINED_MD,
                )
            st.success("Extraction complete!")
            st.session_state.agent = build_agent()

    st.markdown("---")
    with st.expander("📚 PDF Pages", expanded=True):
        pdf_images = sorted(Path(IMAGES_FOLDER).glob("*.png"))
        if not pdf_images:
            st.info("No PDF pages to display yet.")
        else:
            for img in pdf_images:
                st.image(str(img), use_column_width=True)

# --- STATE INIT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": (
             "🤖 Hello! How can I assist you today?\n"
             "💡 Tips:\n"
             "- First upload Your article PDF and Run Extraction\n"
             "- Type '🚪 exit' to leave\n"
             "- Type '📝 summary' to summarize"
         )}
    ]
    st.session_state.pending = False
    st.session_state.agent = None

# --- TOP-LEVEL INPUT ---
user_input = st.chat_input("Type 'summary', 'exit', or your question…")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 1) Did the assistant just ask “save this summary as a PDF”?
    prev = (
        st.session_state.chat_history[-2]["content"].lower()
        if len(st.session_state.chat_history) >= 2 else ""
    )
    if "save this summary as a pdf" in prev:
        if user_input.strip().lower() in ("yes", "y"):
            save_summary_pdf(SUMMARY_MD, SUMMARY_PDF)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ Saved the summary as {SUMMARY_PDF}"
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "👌 Okay, I won’t save the summary as a PDF."
            })
        st.experimental_rerun()

    # 2) Exit command?
    if user_input.strip().lower() in ("exit", "quit", "escape", "🚪 exit"):
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "👋 Goodbye!"}
        )
        st.experimental_rerun()

    # 3) Otherwise, send to the agent
    if st.session_state.agent is None:
        st.warning("Please upload a PDF and run extraction first.")
    else:
        st.session_state.pending = True

    st.experimental_rerun()

# --- VIEW SWITCHER ---
view = st.radio("", ["Chat", "Rendered Summary", "Cropped Images"], horizontal=True)

# --- CHAT TAB (full-width, auto-scroll) ---
if view == "Chat":
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("\n", "  \n"))

# --- RENDERED SUMMARY TAB ---
elif view == "Rendered Summary":
    if Path(SUMMARY_MD).exists():
        md = Path(SUMMARY_MD).read_text(encoding="utf-8")
        html = markdown2.markdown(md, extras=["fenced-code-blocks", "tables"])
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No summary yet. Ask ‘summary’ in chat to generate one.")

# --- CROPPED IMAGES TAB ---
else:
    desc_file = Path(CROPPED_FOLDER) / "descriptions.json"
    if desc_file.exists():
        desc = json.loads(desc_file.read_text())
        for name, txt in desc.items():
            img_path = Path(CROPPED_FOLDER) / name
            if img_path.exists():
                st.image(str(img_path), use_column_width=True)
                st.caption(txt)
                st.markdown("---")
    else:
        st.info("No cropped images yet.")

# --- PROCESS PENDING AGENT QUERY WITH SPINNER ---
if st.session_state.pending:
    last_q = st.session_state.chat_history[-1]["content"]
    with st.spinner("Thinking…"):
        try:
            reply = st.session_state.agent.run(last_q)
        except Exception as e:
            reply = f"[Error during QA]: {e}"
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.pending = False
    st.experimental_rerun()



