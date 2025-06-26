import mysql.connector as mysql
import pandas as pd
import os

DB_CONFIG = {
    "host": "10.10.11.242",
    "user": "omar2",
    "password": "Omar_54321",
    "database": "RME_TEST"
}
TABLE_NAME = "receipts_4_Report"

# Connect to the database
conn = mysql.connect(**DB_CONFIG)

# Query for unique project names and codes
query = f"""
    SELECT DISTINCT RECEIPT_PRJ_CODE, RECEIPT_PRJ_NAME
    FROM {TABLE_NAME}
    WHERE RECEIPT_PRJ_NAME IS NOT NULL
    ORDER BY RECEIPT_PRJ_CODE
"""

df = pd.read_sql(query, conn)

# Save to Excel
output_path = os.path.join(os.path.dirname(__file__), 'unique_receipt_projects.xlsx')
df.to_excel(output_path, index=False)
print(f"Saved unique receipt projects to {output_path}")

conn.close() 