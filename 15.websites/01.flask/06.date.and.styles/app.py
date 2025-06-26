from flask import Flask, render_template, send_file, request
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
OUTPUT_DIR = r"D:\site\rev07\downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Route to load dropdown lists for Projects and Clients

from datetime import datetime 

@app.route('/')
def index():
    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        # Fetch and clean project names
        query_projects = f"SELECT DISTINCT TRX_PRJ_NAME FROM {TABLE_NAME} WHERE TRX_PRJ_NAME IS NOT NULL"
        projects_df = pd.read_sql(query_projects, engine)
        project_names = projects_df['TRX_PRJ_NAME'].astype(str).str.strip().dropna().unique().tolist()
        project_names.sort()

        # Fetch and clean client names
        query_clients = f"SELECT DISTINCT CUSTOMER_NAME FROM {TABLE_NAME} WHERE CUSTOMER_NAME IS NOT NULL"
        clients_df = pd.read_sql(query_clients, engine)
        client_names = clients_df['CUSTOMER_NAME'].astype(str).str.strip().dropna().unique().tolist()
        client_names.sort()

        # 👇 Pass current year explicitly
        return render_template('index.html', projects=project_names, clients=client_names, current_year=datetime.now().year)

    except Exception as e:
        print("❌ Error fetching dropdown data:", e)
        return render_template('index.html', projects=[], clients=[], current_year=datetime.now().year)

import traceback

def fetch_data(project_name, client_name, from_date=None, to_date=None):
    """Fetches data based on filters and exports a formatted Excel report."""
    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        query = f"""
            SELECT 
                CAST(RECEIPT_ID AS CHAR) AS RECEIPT_ID,  
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
            WHERE 1=1
        """
        params = []

        # Apply filters
        if project_name != "all":
            query += " AND TRX_PRJ_NAME = %s"
            params.append(project_name)

        if client_name != "all":
            query += " AND CUSTOMER_NAME = %s"
            params.append(client_name)

        if from_date:
            query += " AND RECEIPT_DATE >= %s"
            params.append(from_date)

        if to_date:
            query += " AND RECEIPT_DATE <= %s"
            params.append(to_date)

        print(f"🔍 Query:\n{query}")
        print(f"📌 Params: {params}")

        df = pd.read_sql(query, engine, params=tuple(params))

        if df.empty:
            print("⚠️ No data found for the given filters.")
            return None

        # Rename columns for Excel
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

        # Format Date
        df["RPT_Date"] = pd.to_datetime(df["RPT_Date"], errors="coerce").dt.strftime("%d-%b-%y")

        # Save to Excel
        file_name = f"Receipts_Report_{project_name}_{client_name}_{from_date or 'start'}_{to_date or 'end'}.xlsx".replace(" ", "_")
        file_path = os.path.join(OUTPUT_DIR, file_name)
        print(f"📁 Saving Excel file: {file_path}")

        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Receipts Report")
            workbook = writer.book
            worksheet = writer.sheets["Receipts Report"]

            # Excel table range
            last_row = len(df) + 1
            last_col = len(df.columns) - 1
            table_range = f"A1:{chr(65 + last_col)}{last_row}"

            # Add formatted table
            worksheet.add_table(table_range, {
                "columns": [{"header": col} for col in df.columns],
                "style": "Table Style Medium 9"
            })

            # Column formatting
            number_format = workbook.add_format({"num_format": "#,##0"})
            date_format = workbook.add_format({"num_format": "DD-MMM-YY"})
            text_format = workbook.add_format({"num_format": "@"})

            for col_num, col_name in enumerate(df.columns):
                col_width = max(len(col_name), 10)
                if col_name == "RPT_ ID":
                    worksheet.set_column(col_num, col_num, col_width, text_format)
                elif col_name == "RPT_Date":
                    worksheet.set_column(col_num, col_num, col_width, date_format)
                elif pd.api.types.is_numeric_dtype(df[col_name]):
                    worksheet.set_column(col_num, col_num, col_width, number_format)
                else:
                    worksheet.set_column(col_num, col_num, col_width)

            worksheet.freeze_panes(1, 0)

        print("✅ Excel report generated successfully.")
        return file_path

    except Exception as e:
        print("❌ Error generating report:")
        print(traceback.format_exc())
        return None


# Route to download filtered Excel
@app.route('/download', methods=['POST'])
def download_excel():
    project_name = request.form.get("project_name", "all")
    client_name = request.form.get("client_name", "all")
    from_date = request.form.get("from_date") or None
    to_date = request.form.get("to_date") or None

    file_path = fetch_data(project_name, client_name, from_date, to_date)
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)

    return "Error: Could not generate the report.", 500


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
