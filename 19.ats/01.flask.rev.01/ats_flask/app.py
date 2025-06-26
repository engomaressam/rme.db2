import os
import re
import pandas as pd
import pytesseract
from flask import Flask, render_template, request, send_file
from pdf2image import convert_from_path
from PIL import Image
from werkzeug.utils import secure_filename

# Config
UPLOAD_FOLDER = "uploads"
POPPLER_PATH = r"C:\poppler\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional: if not in PATH
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Skill Keywords
SKILLS = [
    "Primavera", "MS Project", "Power BI", "Excel", "AutoCAD", "Revit",
    "ETABS", "SAP2000", "Safe", "CSI Column", "Civil 3D", "Synchro", "Navisworks",
    "Quantity Surveying", "Cost Estimation", "Claims", "Invoicing", "Baseline",
    "Delay Analysis", "S-Curve", "Cash Flow", "Earned Value", "Resource Histogram",
    "PMP", "PRMG", "Planning Diploma", "Cost Control Workshop", "CAFT"
]

# Functions
def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    all_text = ""
    for i, page in enumerate(pages):
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"temp_{i}.png")
        page.save(image_path, "PNG")
        text = pytesseract.image_to_string(Image.open(image_path))
        all_text += f"\n{text}"
        os.remove(image_path)
    return all_text

def parse_resume_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name = lines[0] if lines else None

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'(\+20|\b0)?[\s\(]*1[0-9]{2}[\s\)-]*[0-9]{3}[\s\-]*[0-9]{4}', text)
    linkedin_match = re.search(r'(https?://)?(www\.)?linkedin\.com/in/\S+', text)
    grad_year_match = re.findall(r'\b(19[9][0-9]|20[0-2][0-9]|2024)\b', text)

    university = next((line for line in lines if "university" in line.lower()), None)
    grad_year = grad_year_match[0] if grad_year_match else None

    return {
        "Name": name,
        "Email": email_match.group() if email_match else None,
        "Phone": phone_match.group() if phone_match else None,
        "LinkedIn": linkedin_match.group() if linkedin_match else None,
        "Graduation Year": grad_year,
        "University": university
    }

def extract_skills(text, skill_list):
    found_skills = set()
    for skill in skill_list:
        if re.search(re.escape(skill), text, re.IGNORECASE):
            found_skills.add(skill)
    return sorted(found_skills)

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        files = request.files.getlist("pdfs")
        for file in files:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            text = extract_text_from_pdf(save_path)
            info = parse_resume_text(text)
            skills = extract_skills(text, SKILLS)
            info["Skills"] = ", ".join(skills)

            results.append(info)
            os.remove(save_path)

        # Export to Excel
        df = pd.DataFrame(results)
        excel_path = os.path.join(app.config["UPLOAD_FOLDER"], "Parsed_CVs.xlsx")
        df.to_excel(excel_path, index=False)

        return render_template("index.html", results=results, download_link="Parsed_CVs.xlsx")

    return render_template("index.html", results=None)

@app.route("/download")
def download():
    path = os.path.join(app.config["UPLOAD_FOLDER"], "Parsed_CVs.xlsx")
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)

