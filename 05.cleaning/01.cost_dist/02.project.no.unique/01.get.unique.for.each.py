import os
import pandas as pd

# Paste your file path from File Explorer as a string here
file_path_from_explorer = r'D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\01.tables\01.cost.dist\02.xlsx'

# Normalize the path
directory = os.path.normpath(file_path_from_explorer)

# List all the Excel files in the directory
excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

# Initialize empty DataFrames to store unique values
unique_project_no_df = pd.DataFrame(columns=['project_no.unique'])
unique_project_name_df = pd.DataFrame(columns=['project_name.unique'])
unique_expend_org_df = pd.DataFrame(columns=['expend_org.unique'])

# Loop through each Excel file
for excel_file in excel_files:
    # Construct the full file path
    file_path = os.path.join(directory, excel_file)
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    
    # Get unique values for specified columns
    unique_project_no = df['project_no'].unique()
    unique_project_name = df['project_name'].unique()
    unique_expend_org = df['expend_org'].unique()
    
    # Append unique values to the DataFrames
    unique_project_no_df = pd.concat([unique_project_no_df, pd.DataFrame({'project_no.unique': unique_project_no})], ignore_index=True)
    unique_project_name_df = pd.concat([unique_project_name_df, pd.DataFrame({'project_name.unique': unique_project_name})], ignore_index=True)
    unique_expend_org_df = pd.concat([unique_expend_org_df, pd.DataFrame({'expend_org.unique': unique_expend_org})], ignore_index=True)

# Construct the output file names
output_project_no_file = 'unique_project_no.xlsx'
output_project_name_file = 'unique_project_name.xlsx'
output_expend_org_file = 'unique_expend_org.xlsx'

# Save the unique DataFrames to new Excel files
unique_project_no_df.to_excel(os.path.join(directory, output_project_no_file), index=False)
unique_project_name_df.to_excel(os.path.join(directory, output_project_name_file), index=False)
unique_expend_org_df.to_excel(os.path.join(directory, output_expend_org_file), index=False)