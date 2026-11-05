"""Text of a PDF submission, for extraction/authentic_extractor.py.

The rest of the original file was a prototype parser for an external PDF dataset; it was never part
of this corpus's extraction and is in docs/archive/code/extraction/pdf_extractor.py (D-205).
"""
import PyPDF2


def extract_pdf_text(pdf_path):
    """Extract text content from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""
