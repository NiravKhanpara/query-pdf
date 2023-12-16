import pytesseract
from pdf2image import convert_from_path


def get_text_from_scanned_pdf(pdf_path):
    text = ''
    images = convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text
