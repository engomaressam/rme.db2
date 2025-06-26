import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas as pd
import os

# ====== OCR Function ======
def extract_text_from_pdf(pdf_path, poppler_path=None):
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    all_text = ""
    for i, page in enumerate(pages):
        image_path = f"page_{i}.png"
        page.save(image_path, "PNG")
        text = pytesseract.image_to_string(Image.open(image_path))
        all_text += f"\n{text}"
        os.remove(image_path)
    return all_text

# ====== Resume Parser ======
def parse_resume_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name = lines[0] if lines else None

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group() if email_match else None

    phone_match = re.search(r'(\+20|\b0)?[\s\(]*1[0-9]{2}[\s\)-]*[0-9]{3}[\s\-]*[0-9]{4}', text)
    phone = phone_match.group() if phone_match else None

    linkedin_match = re.search(r'(https?://)?(www\.)?linkedin\.com/in/\S+', text)
    linkedin = linkedin_match.group() if linkedin_match else None

    grad_year_match = re.findall(r'\b(19[9][0-9]|20[0-2][0-9]|2024)\b', text)
    grad_year = grad_year_match[0] if grad_year_match else None

    university = None
    for line in lines:
        if "university" in line.lower():
            university = line
            break

    return {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "LinkedIn": linkedin,
        "Graduation Year": grad_year,
        "University": university
    }

# ====== Skill Extractor ======
SKILLS = [
    "Primavera", "MS Project", "Power BI", "Excel", "AutoCAD", "Revit",
    "ETABS", "SAP2000", "Safe", "CSI Column", "Civil 3D", "Synchro", "Navisworks",
    "Quantity Surveying", "Cost Estimation", "Claims", "Invoicing", "Baseline",
    "Delay Analysis", "S-Curve", "Cash Flow", "Earned Value", "Resource Histogram",
    "PMP", "PRMG", "Planning Diploma", "Cost Control Workshop", "CAFT"
]

def extract_skills(text, skill_list):
    found_skills = set()
    for skill in skill_list:
        if re.search(re.escape(skill), text, re.IGNORECASE):
            found_skills.add(skill)
    return sorted(found_skills)

# ====== Main Program ======
if __name__ == "__main__":
    Tk().withdraw()
    pdf_files = askopenfilenames(filetypes=[("PDF files", "*.pdf")])

    all_results = []

    if pdf_files:
        for pdf_file in pdf_files:
            print(f"\n📄 Processing: {os.path.basename(pdf_file)}")
            text = extract_text_from_pdf(pdf_file, poppler_path=r"C:\poppler\Library\bin")

            structured_info = parse_resume_text(text)
            skills = extract_skills(text, SKILLS)
            structured_info["Skills"] = ", ".join(skills)

            all_results.append(structured_info)

            print("--- STRUCTURED INFO ---")
            for key, value in structured_info.items():
                print(f"{key}: {value if value else 'Not Found'}")

        # ✅ Save once at the end
        df = pd.DataFrame(all_results)
        df.to_excel("Parsed_CVs.xlsx", index=False)
        print("\n✅ All results exported to 'Parsed_CVs.xlsx'")

    else:
        print("No files selected.")
