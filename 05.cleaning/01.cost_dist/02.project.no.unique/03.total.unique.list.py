import os
import pandas as pd

# Paste your file path from File Explorer as a string here
file_path_from_explorer = r'D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\01.tables\01.cost.dist\02.xlsx'

# Normalize the path
directory = os.path.normpath(file_path_from_explorer)

# List all the Excel files in the directory
excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

# Create an empty list to store unique project ids
unique_projects = []

# Loop through each Excel file
for excel_file in excel_files:
    # Construct the full file path
    file_path = os.path.join(directory, excel_file)
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    # Get unique project ids for "project_no" and "project_name" columns
    unique_project_identifiers = df[['project_no', 'project_name']].drop_duplicates()
    
    # Append the unique project ids to the list
    unique_projects.append(unique_project_identifiers)

# Concatenate all unique project ids into a single DataFrame
unique_projects_df = pd.concat(unique_projects, ignore_index=True)

# Get unique project numbers and project names from the appended DataFrame
unique_project_numbers = unique_projects_df['project_no'].unique()
unique_project_names = unique_projects_df['project_name'].unique()

# Create a DataFrame for the unique project numbers and project names
unique_list_df = pd.DataFrame({'project_no': unique_project_numbers, 'project_name': unique_project_names})

# Save the unique project ids to a new Excel file
output_file = 'unique_projects.xlsx'
unique_list_df.to_excel(os.path.join(directory, output_file), index=False)