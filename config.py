# config.py
from pathlib import Path

# Project root: two levels up from this config file
ROOT = Path(__file__).resolve().parent

# PDF & extraction paths
PDF_PATH       = ROOT / "media" / "uploads" / "input.pdf"
IMAGES_FOLDER  = ROOT / "media" / "pdf_images"
CROPPED_FOLDER = ROOT / "media" / "crops"
COMBINED_MD    = ROOT / "media" / "combined_extraction.md"

# Summary outputs
SUMMARY_MD  = ROOT / "media" / "formatted_summary.md"
SUMMARY_PDF = ROOT / "media" / "formatted_summary.pdf"

# Chroma DB
CHROMA_DB_DIR = ROOT / "media" / "chroma_db"
