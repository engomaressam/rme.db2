import pdfplumber
import pandas as pd
import re
import os
import sys

# Redirect stdout and stderr to a log file
sys.stdout = open('verification_results.txt', 'w')
sys.stderr = sys.stdout

# Define input file paths
input_pdf_path = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae\ADWEA_APPROVED_CONTRACTORS_LIST.pdf"
input_excel_path = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae\ADWEA_APPROVED_CONTRACTORS_LIST.xlsx"

# Page numbers to verify (1-indexed)
pages_to_verify = [3, 13, 23]

# Function to extract Work Group and Criteria from page text (from previous script)
def extract_pdf_headers(page_text):
    work_group_match = re.search(r'Work Group:\s*(.*?)(?:\n|Criteria)', page_text, re.DOTALL)
    criteria_match = re.search(r'Criteria\s*:\s*(.*?)(?:\n|Company ID)', page_text, re.DOTALL)
    work_group = work_group_match.group(1).strip().replace('\n', ' ') if work_group_match else "PDF_WG_Not_Found"
    criteria = criteria_match.group(1).strip().replace('\n', ' ') if criteria_match else "PDF_Criteria_Not_Found"
    return work_group, criteria

def find_first_table_and_first_row(page):
    tables = page.extract_tables({
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
    })
    for table in tables:
        if not table or len(table) < 2: continue # Skip empty or header-only tables
        header_row_idx = -1
        for i, row in enumerate(table):
            row_str = " ".join(filter(None, [str(c) for c in row]))
            if 'Company ID' in row_str and 'Company Name' in row_str:
                header_row_idx = i
                break
        if header_row_idx != -1 and len(table) > header_row_idx + 1:
            pdf_headers = [str(h).replace('\n', ' ').strip() if h else f'Unnamed_{k}' for k, h in enumerate(table[header_row_idx])]
            first_data_row = table[header_row_idx + 1]
            if any(cell and str(cell).strip() for cell in first_data_row):
                # Try to find Company ID and Company Name columns
                company_id_col_idx = next((i for i, h in enumerate(pdf_headers) if 'Company ID' in h), None)
                company_name_col_idx = next((i for i, h in enumerate(pdf_headers) if 'Company Name' in h), None)
                
                pdf_company_id = str(first_data_row[company_id_col_idx]).strip() if company_id_col_idx is not None and company_id_col_idx < len(first_data_row) else None
                pdf_company_name = str(first_data_row[company_name_col_idx]).strip() if company_name_col_idx is not None and company_name_col_idx < len(first_data_row) else None
                return pdf_company_id, pdf_company_name
    return None, None

print(f"Verifying Excel data against PDF for pages: {pages_to_verify}\n")

# Load Excel data
try:
    excel_df = pd.read_excel(input_excel_path)
    print(f"Successfully loaded Excel file: {input_excel_path}")
    # Standardize column names in Excel for comparison (e.g., remove extra spaces, normalize case)
    excel_df.columns = [str(col).strip() for col in excel_df.columns]
except FileNotFoundError:
    print(f"Error: Excel file not found at {input_excel_path}")
    exit()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Open PDF
try:
    with pdfplumber.open(input_pdf_path) as pdf:
        print(f"Successfully opened PDF: {input_pdf_path}\n")
        for page_num_1_indexed in pages_to_verify:
            print(f"--- Verifying Page {page_num_1_indexed} ---")
            page_num_0_indexed = page_num_1_indexed - 1
            if page_num_0_indexed < 0 or page_num_0_indexed >= len(pdf.pages):
                print(f"  Error: Page number {page_num_1_indexed} is out of range for PDF.")
                continue
            
            page = pdf.pages[page_num_0_indexed]
            page_text = page.extract_text()
            if not page_text:
                print("  Error: No text found on this PDF page.")
                continue

            pdf_work_group, pdf_criteria = extract_pdf_headers(page_text)
            print(f"  PDF Headers - Work Group: '{pdf_work_group}', Criteria: '{pdf_criteria}'")

            pdf_company_id, pdf_company_name = find_first_table_and_first_row(page)
            if not pdf_company_id and not pdf_company_name:
                print("  Error: Could not extract Company ID or Name from the first data row of the table on this PDF page.")
                continue
            print(f"  PDF First Row - Company ID: '{pdf_company_id}', Company Name: '{pdf_company_name}'")

            # Find corresponding row in Excel
            matched_excel_row = None

            # Prepare string versions of relevant Excel columns for robust matching
            # This also handles cases where columns might be missing or have mixed types.
            excel_company_id_col_present = 'Company ID' in excel_df.columns
            excel_company_name_col_present = 'Company Name' in excel_df.columns
            excel_work_group_col_present = 'Work Group' in excel_df.columns
            excel_criteria_col_present = 'Criteria' in excel_df.columns

            temp_excel_df = excel_df.copy() # Work on a copy to avoid modifying original excel_df with temp columns

            if excel_company_id_col_present:
                temp_excel_df['Company ID_str_match'] = temp_excel_df['Company ID'].astype(str).str.strip()
            if excel_company_name_col_present:
                temp_excel_df['Company Name_str_match'] = temp_excel_df['Company Name'].astype(str).str.strip()
            if excel_work_group_col_present:
                temp_excel_df['Work Group_str_match'] = temp_excel_df['Work Group'].astype(str).str.strip()
            else:
                print(f"  Warning: 'Work Group' column not found in Excel for page {page_num_1_indexed}. Cannot perform exact match on Work Group.")
            if excel_criteria_col_present:
                temp_excel_df['Criteria_str_match'] = temp_excel_df['Criteria'].astype(str).str.strip()
            else:
                print(f"  Warning: 'Criteria' column not found in Excel for page {page_num_1_indexed}. Cannot perform exact match on Criteria.")

            # Attempt 1: Match by Company ID, Work Group, and Criteria
            if pdf_company_id and excel_company_id_col_present and excel_work_group_col_present and excel_criteria_col_present:
                exact_matches_id = temp_excel_df[
                    (temp_excel_df['Company ID_str_match'] == str(pdf_company_id).strip()) &
                    (temp_excel_df['Work Group_str_match'] == str(pdf_work_group).strip()) &
                    (temp_excel_df['Criteria_str_match'] == str(pdf_criteria).strip())
                ]
                if not exact_matches_id.empty:
                    if len(exact_matches_id) > 1:
                        print(f"  Info: Multiple exact matches found in Excel using Company ID for Page {page_num_1_indexed}, Company ID '{pdf_company_id}'. Using the first one.")
                    matched_excel_row = exact_matches_id.iloc[0]
            
            # Attempt 2: If no exact match by ID, fallback to Company Name, Work Group, and Criteria
            if matched_excel_row is None and pdf_company_name and excel_company_name_col_present and excel_work_group_col_present and excel_criteria_col_present:
                exact_matches_name = temp_excel_df[
                    (temp_excel_df['Company Name_str_match'] == str(pdf_company_name).strip()) &
                    (temp_excel_df['Work Group_str_match'] == str(pdf_work_group).strip()) &
                    (temp_excel_df['Criteria_str_match'] == str(pdf_criteria).strip())
                ]
                if not exact_matches_name.empty:
                    if len(exact_matches_name) > 1:
                        print(f"  Info: Multiple exact matches found in Excel using Company Name for Page {page_num_1_indexed}, Company Name '{pdf_company_name}'. Using the first one.")
                    matched_excel_row = exact_matches_name.iloc[0]

            # If still no exact match after specific attempts, you could add broader fallbacks here if desired
            # For now, if matched_excel_row is None, it means the specific combination was not found.
            
            if matched_excel_row is not None:
                excel_work_group = str(matched_excel_row.get('Work Group', 'Excel_WG_Not_Found')).strip()
                excel_criteria = str(matched_excel_row.get('Criteria', 'Excel_Criteria_Not_Found')).strip()
                print(f"  Excel Match - Work Group: '{excel_work_group}', Criteria: '{excel_criteria}'")

                if pdf_work_group == excel_work_group:
                    print("  Verification: Work Group MATCHES.")
                else:
                    print("  Verification: Work Group MISMATCH.")
                
                if pdf_criteria == excel_criteria:
                    print("  Verification: Criteria MATCHES.")
                else:
                    print("  Verification: Criteria MISMATCH.")
            else:
                print(f"  Error: No matching row found in Excel for Company ID '{pdf_company_id}' or Company Name '{pdf_company_name}'.")
            print("---")

except FileNotFoundError:
    print(f"Error: PDF file not found at {input_pdf_path}")
except Exception as e:
    print(f"An error occurred during PDF processing: {e}")

print("Verification script finished.")

# Close the output file
if sys.stdout != sys.__stdout__:
    sys.stdout.close()
    sys.stdout = sys.__stdout__ # Restore standard output
    sys.stderr = sys.__stderr__ # Restore standard error
