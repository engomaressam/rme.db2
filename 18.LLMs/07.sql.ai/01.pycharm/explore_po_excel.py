import pandas as pd
import numpy as np
import os
import sys

# Install required packages if not available
try:
    import openpyxl
except ImportError:
    print("Installing missing dependency: openpyxl")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl installed successfully.")

# Define the path to the Excel file
excel_path = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\07.sql.ai\01.erp.table\RME_PO_Follow_up_180625.xlsx"

# Check if the file exists
if not os.path.exists(excel_path):
    print(f"Error: The file {excel_path} does not exist.")
    exit(1)

# Read the Excel file
try:
    df = pd.read_excel(excel_path)
    print(f"Successfully read Excel file with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Display column names
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    # Display data types for each column
    print("\nColumn data types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    # Display sample data (first 5 rows)
    print("\nSample data (first 5 rows):")
    print(df.head(5))

    # Get some statistics about numeric columns
    print("\nNumeric columns statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found.")

    # Check for null values
    print("\nNull values count per column:")
    print(df.isnull().sum())

except Exception as e:
    print(f"Error reading Excel file: {e}")

print("\nExploration complete.")
