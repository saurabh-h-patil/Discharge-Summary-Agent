"""
PDF Ingestion Tool — converts scanned PDF pages to images and extracts text via GPT-4o vision.
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from app.core.llm import get_llm_client
from app.core.prompts import VISION_EXTRACTION_PROMPT, DOCUMENT_CLASSIFIER_PROMPT
from app.models.clinical import ExtractedPage, ExtractedDocument, DocumentType


def pdf_to_page_images(pdf_path: str, dpi: int = 200) -> list[tuple[int, str]]:
    """
    Convert a PDF to a list of base64-encoded PNG images (one per page).
    
    Returns list of (page_number, base64_image_string).
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append((page_num + 1, img_b64))  # 1-indexed

    doc.close()
    return images


def extract_text_from_page(
    page_image_b64: str,
    page_number: int,
    source_file: str,
) -> ExtractedPage:
    """
    Use GPT-4o vision to extract text from a single scanned page image.
    """
    llm = get_llm_client()

    try:
        raw_text = llm.vision(
            image_base64=page_image_b64,
            prompt=VISION_EXTRACTION_PROMPT,
        )
    except Exception as e:
        # Robust failure: don't crash, report the failure
        raw_text = f"[EXTRACTION FAILED: {str(e)}]"
        return ExtractedPage(
            page_number=page_number,
            source_file=source_file,
            raw_text=raw_text,
            is_blank=False,
            extraction_confidence="failed",
        )

    # Check if page is blank
    is_blank = raw_text.strip() in ["[BLANK PAGE]", "[ILLEGIBLE]", ""]

    return ExtractedPage(
        page_number=page_number,
        source_file=source_file,
        raw_text=raw_text,
        is_blank=is_blank,
        extraction_confidence="high" if not is_blank else "low",
    )


def classify_document(text: str) -> DocumentType:
    """
    Use LLM to classify a document type from its extracted text.
    """
    llm = get_llm_client()

    try:
        result = llm.chat(
            messages=[
                {"role": "system", "content": "You are a clinical document classifier."},
                {"role": "user", "content": DOCUMENT_CLASSIFIER_PROMPT + f"\n\nDocument text:\n{text[:3000]}"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        # Parse the result
        result = result.strip().upper().replace(" ", "_")
        try:
            return DocumentType(result)
        except ValueError:
            return DocumentType.OTHER
    except Exception:
        return DocumentType.OTHER


def ingest_pdf(
    pdf_path: str,
    batch_size: int = 5,
    progress_callback=None,
) -> list[ExtractedPage]:
    """
    Full PDF ingestion pipeline:
    1. Convert PDF to page images
    2. Extract text from each page via vision
    3. Return list of ExtractedPage objects
    
    Uses batching to manage API rate limits.
    """
    source_file = os.path.basename(pdf_path)
    print(f"\n📄 Ingesting PDF: {source_file}")

    # Step 1: Convert to images
    print(f"  Converting to images...")
    page_images = pdf_to_page_images(pdf_path)
    print(f"  Found {len(page_images)} pages")

    # Step 2: Extract text from each page
    extracted_pages = []
    total = len(page_images)

    for i, (page_num, img_b64) in enumerate(page_images):
        print(f"  Extracting page {page_num}/{total}...")

        page = extract_text_from_page(
            page_image_b64=img_b64,
            page_number=page_num,
            source_file=source_file,
        )
        extracted_pages.append(page)

        if progress_callback:
            progress_callback(page_num, total)

    # Report stats
    blank = sum(1 for p in extracted_pages if p.is_blank)
    failed = sum(1 for p in extracted_pages if p.extraction_confidence == "failed")
    print(f"  ✅ Extracted: {total - blank - failed} pages, {blank} blank, {failed} failed")

    return extracted_pages


def group_pages_into_documents(pages: list[ExtractedPage]) -> list[ExtractedDocument]:
    """
    Group extracted pages into logical documents and classify each.
    Uses LLM to classify based on content, then groups consecutive pages of the same type.
    """
    if not pages:
        return []

    print(f"\n📋 Classifying {len(pages)} pages into documents...")

    # Classify each non-blank page
    for page in pages:
        if not page.is_blank and page.extraction_confidence != "failed":
            page.document_type = classify_document(page.raw_text)
            print(f"  Page {page.page_number}: {page.document_type.value}")

    # Group consecutive pages of the same type into documents
    documents = []
    current_pages = []
    current_type = None

    for page in pages:
        if page.is_blank or page.extraction_confidence == "failed":
            continue

        if current_type is None or page.document_type == current_type:
            current_pages.append(page)
            current_type = page.document_type
        else:
            # Save current document
            if current_pages:
                doc = ExtractedDocument(
                    doc_id=f"doc_{len(documents) + 1}",
                    document_type=current_type,
                    source_file=current_pages[0].source_file,
                    pages=current_pages,
                )
                doc.full_text = doc.get_full_text()
                documents.append(doc)
            # Start new document
            current_pages = [page]
            current_type = page.document_type

    # Don't forget the last group
    if current_pages and current_type:
        doc = ExtractedDocument(
            doc_id=f"doc_{len(documents) + 1}",
            document_type=current_type,
            source_file=current_pages[0].source_file,
            pages=current_pages,
        )
        doc.full_text = doc.get_full_text()
        documents.append(doc)

    print(f"  ✅ Grouped into {len(documents)} documents")
    for doc in documents:
        print(f"    - {doc.doc_id}: {doc.document_type.value} ({len(doc.pages)} pages)")

    return documents
