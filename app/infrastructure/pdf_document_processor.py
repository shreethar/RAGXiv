import fitz

from app.infrastructure.document_processor import DocumentProcessor


class PdfDocumentProcessor(DocumentProcessor):
    def extract_text(self, *, document: bytes) -> str:
        if not document:
            return ""

        pdf = fitz.open(stream=document, filetype="pdf")

        try:
            pages = [
                page.get_text()
                for page in pdf
            ]

            return "\n".join(pages)
        finally:
            pdf.close()