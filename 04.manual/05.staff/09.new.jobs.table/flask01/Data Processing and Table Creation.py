import pandas as pd

# Read data from Excel file
data_df = pd.read_excel("data/staff.xlsx")

# Function to group data by project and create tables
def create_project_tables(data):
    tables = {}
    for project in data["Project"].unique():
        project_data = data[data["Project"] == project]

        # Get unique jobs and months
        unique_jobs = list(project_data["Job"].unique())
        months = list(project_data["Month"].unique())

        # Create a dictionary to store job counts for each month
        job_counts = {}
        for job in unique_jobs:
            job_counts[job] = [0] * len(months)
            for index, row in project_data.iterrows():
                if row["Job"] == job and row["Month"] in months:
                    job_counts[job][months.index(row["Month"])] += 1

        # Create a DataFrame for the table
        table_df = pd.DataFrame({"Job": unique_jobs})
        table_df["Month"] = months
        for job, counts in job_counts.items():
            table_df[job] = counts

        # Convert DataFrame to HTML table
        html_table = table_df.to_html()
        tables[project] = html_table

    return tables

# Create tables for all projects
project_tables = create_project_tables(data_df.copy())

# Combine tables into a single string (optional)
combined_tables = "\n".join(project_tables.values())
