from docx import Document
import os

def read_word_doc(docx_path):
    if not os.path.exists(docx_path):
        print(f"Word document not found: {docx_path}")
        return
        
    print(f"Reading {docx_path}...")
    doc = Document(docx_path)
    
    # Save paragraphs to text file
    output_path = docx_path + '.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                f.write(f"[{i}] {text}\n\n")
    print(f"\nSaved content to {output_path}")

if __name__ == "__main__":
    docx_path = os.path.join("reports", "Top_20_Documents_Summary.docx")
    read_word_doc(docx_path)
