from flask import Flask, render_template, send_file
import mysql.connector as mysql
import pandas as pd
import os
from sqlalchemy import create_engine

app = Flask(__name__)

# Database connection details
DB_CONFIG = {
    "host": "10.10.11.242",
    "user": "omar2",
    "password": "Omar_54321",
    "database": "RME_TEST"
}
TABLE_NAME = "receipts_4_Report"

# Directory to store Excel files
OUTPUT_DIR = r"C:\PythonScripts\flask-excel\downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from sqlalchemy import create_engine
import pandas as pd
import os

def fetch_data():
    """Fetches only the required columns from MySQL and saves it as an Excel file with renamed headers."""
    try:
        # Convert MySQL connection details into an SQLAlchemy connection string
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        # Select only the required fields
        query = f"""
            SELECT 
                RECEIPT_ID, 
                RECEIPT_NUMBER, 
                CUSTOMER_NO, 
                CUSTOMER_NAME, 
                RECEIPT_PRJ_CODE, 
                RECEIPT_PRJ_NAME, 
                TRX_PRJ_CODE, 
                TRX_PRJ_NAME, 
                RECEIPT_DATE, 
                INV_NUM AS INVOICE_NO, 
                TRANSACTION_AMOUNT, 
                TAX_AMOUNT, 
                TOTAL_AFTER_TAX, 
                NEW_CALCULATED_TOTAL_ADJ AS DEDUCTIONS_ADJUSTMENTS, 
                CALCULATED_AMOUNT_TO_COLLECT, 
                STATUS AS APPLICATION_STATUS, 
                RECEIPT_AMOUNT, 
                AMOUNT_APPLIED, 
                ATTRIBUTE1 AS COLLECTION_TYPE, 
                TOTAL_AMOUNT_APPLIED, 
                AMOUNT_DUE_REMAINING 
            FROM {TABLE_NAME}
        """
        
        # Read SQL data into DataFrame
        df = pd.read_sql(query, engine)

        # Rename columns to match the required output format
        column_mapping = {
            "RECEIPT_ID": "RPT_ ID",
            "RECEIPT_NUMBER": "RPT_NO",
            "CUSTOMER_NO": "Client No.",
            "CUSTOMER_NAME": "Client Name",
            "RECEIPT_PRJ_NAME": "RPT_Prj. Name",
            "RECEIPT_PRJ_CODE": "RPT_Prj. Code",
            "TRX_PRJ_NAME": "TRX_Prj. Name", 
            "TRX_PRJ_CODE": "TRX_Prj. Code",

            "RECEIPT_DATE": "RPT_Date",
            "INVOICE_NO": "Invoice No.",
            "TRANSACTION_AMOUNT": "Value Of Work",
            "TAX_AMOUNT": "Tax",
            "TOTAL_AFTER_TAX": "VOW After Tax",
            "DEDUCTIONS_ADJUSTMENTS": "Deductions/Adjustments",
            "CALCULATED_AMOUNT_TO_COLLECT": "Net",
            "APPLICATION_STATUS": "Application Status",
            "RECEIPT_AMOUNT": "RPT_Amount",
            "AMOUNT_APPLIED": "Amount Applied",
            "COLLECTION_TYPE": "Collection Type",
            "TOTAL_AMOUNT_APPLIED": "Collected Amount",
            "AMOUNT_DUE_REMAINING": "Variance"
        }
        df.rename(columns=column_mapping, inplace=True)

        # Save to Excel with the new filename "Receipts_Report.xlsx"
        file_path = os.path.join(OUTPUT_DIR, "Receipts_Report.xlsx")
        df.to_excel(file_path, index=False)

        print(f"✅ Report generated successfully: {file_path}")  # Debugging
        return file_path
    except Exception as e:
        print(f"❌ ERROR: {e}")  # Log error details
        return None



@app.route('/')
def home():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/download')
def download_excel():
    """Generate and download the Excel report."""
    file_path = fetch_data()
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Error: Could not generate the report.", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
