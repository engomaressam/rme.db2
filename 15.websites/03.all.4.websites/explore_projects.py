import mysql.connector as mysql
import pandas as pd
import os

DB_CONFIG = {
    "host": "10.10.11.242",
    "user": "omar2",
    "password": "Omar_54321",
    "database": "RME_TEST"
}
TABLE_NAME = "SWD_Collection_Report"

# Connect to the database
conn = mysql.connect(**DB_CONFIG)

# Query for unique project numbers and names
query = f"""
    SELECT DISTINCT PROJECT_NUM, PROJECT_NAME
    FROM {TABLE_NAME}
    WHERE PROJECT_NAME IS NOT NULL
    ORDER BY PROJECT_NUM
"""

df = pd.read_sql(query, conn)

# Save to Excel
output_path = os.path.join(os.path.dirname(__file__), 'unique_projects.xlsx')
df.to_excel(output_path, index=False)
print(f"Saved unique projects to {output_path}")

conn.close() 