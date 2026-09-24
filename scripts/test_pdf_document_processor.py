import fitz
import pytest

from app.infrastructure.pdf_document_processor import (
    PdfDocumentProcessor,
)

@pytest.mark.integration
def test_extract_text_from_academic_paper():
    pdf = fitz.open()

    try:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "RAGxiv: Retrieval-Augmented Generation for Scientific Papers",
        )
        page.insert_text(
            (72, 100),
            "Abstract: This paper investigates retrieval "
            "augmented generation over scientific literature.",
        )

        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "1. Introduction",
        )
        page.insert_text(
            (72, 100),
            "Scientific papers contain valuable information "
            "that can be retrieved and used by language models.",
        )

        document = pdf.tobytes()

    finally:
        pdf.close()

    processor = PdfDocumentProcessor()

    result = processor.extract_text(document=document)

    assert "RAGxiv" in result
    assert "Abstract" in result
    assert "1. Introduction" in result
    assert "Scientific papers" in result

def create_pdf(*pages: str) -> bytes:
    pdf = fitz.open()

    try:
        for text in pages:
            page = pdf.new_page()
            page.insert_text((72, 72), text)

        return pdf.tobytes()
    finally:
        pdf.close()


def test_extract_text_from_empty_document():
    processor = PdfDocumentProcessor()

    result = processor.extract_text(document=b"")

    assert result == ""


def test_extract_text_from_pdf():
    processor = PdfDocumentProcessor()

    document = create_pdf("Hello, RAGxiv!")

    result = processor.extract_text(document=document)

    assert "Hello, RAGxiv!" in result


def test_extract_text_from_multiple_pages():
    processor = PdfDocumentProcessor()

    document = create_pdf(
        "This is page one.",
        "This is page two.",
    )

    result = processor.extract_text(document=document)

    assert "This is page one." in result
    assert "This is page two." in result


def test_extract_text_rejects_invalid_pdf():
    processor = PdfDocumentProcessor()

    with pytest.raises(Exception):
        processor.extract_text(
            document=b"This is not a PDF."
        )