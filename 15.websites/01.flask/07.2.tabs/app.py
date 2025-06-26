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

        # Fetch and sort distinct project names
        query_projects = "SELECT DISTINCT TRX_PRJ_NAME FROM receipts_4_Report WHERE TRX_PRJ_NAME IS NOT NULL"
        projects_df = pd.read_sql(query_projects, engine)
        project_names = projects_df['TRX_PRJ_NAME'].dropna().astype(str).str.strip().tolist()
        project_names.sort()

        # Fetch and sort distinct client names
        query_clients = "SELECT DISTINCT CUSTOMER_NAME FROM receipts_4_Report WHERE CUSTOMER_NAME IS NOT NULL"
        clients_df = pd.read_sql(query_clients, engine)
        client_names = clients_df['CUSTOMER_NAME'].dropna().astype(str).str.strip().tolist()
        client_names.sort()

        return render_template('index.html', projects=project_names, clients=client_names)

    except Exception as e:
        print("Error fetching dropdown data:", e)
        return render_template('index.html', projects=[], clients=[])

import traceback

def fetch_data(project_name, client_name, from_date, to_date):
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

        print(f"🔍 Executing SQL with params: {params}")
        df = pd.read_sql(query, engine, params=tuple(params))  # ✅ FIXED HERE

        if df.empty:
            print("⚠️ No data found.")
            return None

        # Rename columns
        col_map = {
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
        df.rename(columns=col_map, inplace=True)

        df["RPT_Date"] = pd.to_datetime(df["RPT_Date"], errors="coerce")
        file_name = f"Receipts_Report_{project_name}_{client_name}.xlsx".replace(" ", "_")
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # Generate Summary Table (backend-calculated, no formulas)
        summary_df = df[["RPT_Prj. Name", "Invoice No."]].drop_duplicates().copy()
        summary_df["Value Of Work"] = summary_df.apply(
            lambda row: df.loc[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."]),
                "Value Of Work"
            ].max(), axis=1
        )
        summary_df["Net"] = summary_df.apply(
            lambda row: df.loc[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."]),
                "Net"
            ].max(), axis=1
        )
        summary_df["Amount Applied"] = summary_df.apply(
            lambda row: df.loc[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."]),
                "Amount Applied"
            ].sum(), axis=1
        )

        # Write to Excel with formatting
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            # Details Sheet
            df.to_excel(writer, sheet_name="Details", index=False)
            details_ws = writer.sheets["Details"]
            workbook = writer.book

            # Summary Sheet
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            summary_ws = writer.sheets["Summary"]

            # Style settings
            number_format = workbook.add_format({"num_format": "#,##0"})
            date_format = workbook.add_format({"num_format": "DD-MMM-YY"})
            text_format = workbook.add_format({"num_format": "@"})

            # Format Details
            for col_num, col in enumerate(df.columns):
                col_width = max(12, len(col) + 2)
                col_type = df[col].dtype
                if col == "RPT_Date":
                    details_ws.set_column(col_num, col_num, col_width, date_format)
                elif pd.api.types.is_numeric_dtype(col_type) and col != "RPT_ ID":
                    details_ws.set_column(col_num, col_num, col_width, number_format)
                else:
                    details_ws.set_column(col_num, col_num, col_width)

            details_ws.freeze_panes(1, 0)
            summary_ws.freeze_panes(1, 0)

        print(f"✅ Excel report generated with summary: {file_path}")
        return file_path

    except Exception as e:
        import traceback
        print("❌ Error generating Excel:")
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
