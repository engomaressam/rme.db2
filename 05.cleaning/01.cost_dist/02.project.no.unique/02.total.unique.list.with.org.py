# %%
import os
import pandas as pd

# %%
# Paste your file path from File Explorer as a string here
file_path_from_explorer = r'D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\01.tables\01.cost.dist\02.xlsx'

# Normalize the path
directory = os.path.normpath(file_path_from_explorer)

# %%
# List all the Excel files in the directory
excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

# %%
# Initialize an empty list to store unique project identifiers
unique_projects = []

# %%
# Loop through each Excel file
for excel_file in excel_files:
    # Construct the full file path
    file_path = os.path.join(directory, excel_file)
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    # Get unique project identifiers from the DataFrame
    unique_project_identifiers = df[['project_no', 'project_name', 'expend_org']].drop_duplicates()
    
    # Append the unique project identifiers to the list
    unique_projects.append(unique_project_identifiers)

# %%
# Concatenate all unique project identifiers into a single DataFrame
unique_projects_df = pd.concat(unique_projects, ignore_index=True)

# Save the unique project identifiers to a new Excel file
output_file = 'unique_projects.xlsx'
unique_projects_df.to_excel(os.path.join(directory, output_file), index=False)


