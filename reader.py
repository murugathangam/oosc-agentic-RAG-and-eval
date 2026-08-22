import os
from pypdf import PdfReader
from docx import Document

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    else:
        raise ValueError(f"Unsupported file type: {ext} (only .pdf and .docx supported)")


if __name__ == "__main__":
    file_path = input("Enter path to your PDF or Word file: ").strip().strip('"')

    text = extract_text(file_path)

    with open("doc.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extracted {len(text)} characters and saved to doc.txt")