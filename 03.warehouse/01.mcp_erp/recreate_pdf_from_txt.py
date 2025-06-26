from fpdf import FPDF
import os
import re

# Input and output paths
input_txt = "03.warehouse/01.mcp_erp/table_docs/AP_AP_INVOICES_ALL_documentation.txt"
output_pdf = "03.warehouse/01.mcp_erp/table_docs/AP_AP_INVOICES_ALL_documentation_fixed.pdf"
font_path = "03.warehouse/01.mcp_erp/DejaVuSans.ttf"

# Read the text log
with open(input_txt, 'r', encoding='utf-8') as f:
    text = f.read()

# Extract only GPT-4 replies
replies = re.split(r"--- GPT-4 Reply ---", text)
# The first split part is before the first reply, so skip it
reply_texts = [r.strip() for r in replies[1:]]
all_replies = "\n\n".join(reply_texts)

# Create a simple, readable PDF with Unicode font
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_font("DejaVu", "", font_path, uni=True)
pdf.set_font("DejaVu", size=12)

for paragraph in all_replies.split('\n\n'):
    for line in paragraph.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.ln(5)

pdf.output(output_pdf)
print(f"Fixed PDF saved to {output_pdf}") 