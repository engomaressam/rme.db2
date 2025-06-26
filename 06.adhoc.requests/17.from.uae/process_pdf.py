import pdfplumber
import pandas as pd
import re
import os
import sys

# Redirect stdout and stderr to a log file to capture all output
sys.stdout = open('process_output.txt', 'w')
sys.stderr = sys.stdout

# Define input and output file paths
input_pdf = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae\ADWEA_APPROVED_CONTRACTORS_LIST.pdf"
output_excel = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae\ADWEA_APPROVED_CONTRACTORS_LIST.xlsx"

# Function to extract Work Group and Criteria from page text
def extract_headers(page_text, current_page_number):
    # More robust regex
    work_group_match = re.search(r'Work Group:\s*(.*?)(?:\n|Criteria)', page_text, re.DOTALL)
    criteria_match = re.search(r'Criteria\s*:\s*(.*?)(?:\n|Company ID)', page_text, re.DOTALL)
    
    work_group = work_group_match.group(1).strip().replace('\n', ' ') if work_group_match else f"WG_Not_Found_On_Page_{current_page_number}"
    criteria = criteria_match.group(1).strip().replace('\n', ' ') if criteria_match else f"Criteria_Not_Found_On_Page_{current_page_number}"
    
    return work_group, criteria

# Initialize list to store all dataframes
all_tables = []

print(f"Starting PDF processing for: {input_pdf}")

# Check if PDF exists
if not os.path.exists(input_pdf):
    print(f"Error: Input PDF not found at {input_pdf}")
else:
    try:
        with pdfplumber.open(input_pdf) as pdf:
            print(f"PDF has {len(pdf.pages)} pages.")
            for page_num, page in enumerate(pdf.pages):
                print(f"\n--- Processing page {page_num + 1} ---")
                
                text = page.extract_text()
                if not text:
                    print("No text found on this page.")
                    continue
                
                work_group, criteria = extract_headers(text, page_num + 1)
                print(f"Extracted Work Group: '{work_group}'")
                print(f"Extracted Criteria: '{criteria}'")
                
                # Use a specific table extraction strategy
                tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })
                print(f"Found {len(tables)} tables on page {page_num + 1}.")
                
                for i, table in enumerate(tables):
                    print(f"  Processing table {i+1} on page {page_num + 1}")
                    if not table or len(table) < 2:
                        print("    Skipping table: empty or less than 2 rows.")
                        continue
                    
                    header_row_idx = -1
                    for j, row in enumerate(table):
                        row_str = " ".join(filter(None, [str(c) for c in row]))
                        if 'Company ID' in row_str and 'Company Name' in row_str:
                            header_row_idx = j
                            print(f"    Header row found at index {j}: {row}")
                            break
                    
                    if header_row_idx == -1:
                        print("    Skipping table: Header not found.")
                        continue

                    headers = [str(h).replace('\n', ' ').strip() if h else f'Unnamed_{k}' for k, h in enumerate(table[header_row_idx])]
                    data_rows = table[header_row_idx + 1:]
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=headers)
                        df['Work Group'] = work_group
                        df['Criteria'] = criteria
                        all_tables.append(df)
                        print(f"    Successfully processed table, added {len(df)} rows.")

    except Exception as e:
        print(f"An error occurred: {e}")

if all_tables:
    print("\n--- Combining all tables ---")
    try:
        combined_df = pd.concat(all_tables, ignore_index=True)
        
        # Reorder columns
        cols = combined_df.columns.tolist()
        if 'Criteria' in cols:
            cols.insert(0, cols.pop(cols.index('Criteria')))
        if 'Work Group' in cols:
            cols.insert(0, cols.pop(cols.index('Work Group')))
        combined_df = combined_df[cols]

        combined_df = combined_df.dropna(axis=1, how='all').dropna(how='all')
        
        print(f"Saving data to Excel file: {output_excel}")
        combined_df.to_excel(output_excel, index=False)
        print(f"Successfully created Excel file with {len(combined_df)} records.")
        print(f"File saved at: {os.path.abspath(output_excel)}")
    except Exception as e:
        print(f"An error occurred during Excel export: {e}")
else:
    print("\n--- No valid contractor tables were found in the PDF. ---")


