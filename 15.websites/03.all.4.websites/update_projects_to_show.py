import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
projects_to_show_path = os.path.join(base_dir, 'projects.to.show.xlsx')
unique_receipt_projects_path = os.path.join(base_dir, 'unique_receipt_projects.xlsx')

# Read both files
projects_to_show_df = pd.read_excel(projects_to_show_path)
receipt_projects_df = pd.read_excel(unique_receipt_projects_path)

# Ensure column names
projects_to_show_df['PROJECT_NAME'] = projects_to_show_df['PROJECT_NAME'].astype(str).str.strip()
receipt_projects_df['RECEIPT_PRJ_NAME'] = receipt_projects_df['RECEIPT_PRJ_NAME'].astype(str).str.strip()

# Find new projects to add
existing_projects = set(projects_to_show_df['PROJECT_NAME'])
new_projects = [p for p in receipt_projects_df['RECEIPT_PRJ_NAME'] if p not in existing_projects]

# Append new projects
if new_projects:
    new_df = pd.DataFrame({'PROJECT_NAME': new_projects})
    updated_df = pd.concat([projects_to_show_df, new_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset=['PROJECT_NAME']).reset_index(drop=True)
    updated_df.to_excel(projects_to_show_path, index=False)
    print(f"Added {len(new_projects)} new projects. Updated file saved to {projects_to_show_path}")
else:
    print("No new projects to add. File is up to date.") 