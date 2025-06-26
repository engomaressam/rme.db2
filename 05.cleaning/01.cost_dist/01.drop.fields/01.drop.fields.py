import os
import pandas as pd

# Paste your file path from File Explorer as a string here
file_path_from_explorer = r'D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\01.tables\01.cost.dist\08.xlsb.clean\01.test'

# Normalize the path
directory = os.path.normpath(file_path_from_explorer)

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # List of columns to be dropped
        columns_to_drop = [
            "project_name", "project_zone", "project_location",
            "project_floor", "project_area", "expend_org",
            "line_no", "ipc_no", "supplier_name",
            "distributions_status", "distributions_date", "distributions_details"
        ]

        # Drop the specified columns
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Construct the new file name with ".dropped" added
        new_filename = filename.replace('.xlsx', '.dropped.xlsx')

        # Save the modified DataFrame to a new Excel file
        df.to_excel(os.path.join(directory, new_filename), index=False)

        print(f"Processed: {filename} -> Saved as: {new_filename}")

print("Done! All files processed.")