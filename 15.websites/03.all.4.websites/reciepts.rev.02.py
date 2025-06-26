import logging
from flask import Flask, render_template, send_file, request, redirect, url_for, session, flash
import mysql.connector as mysql
import pandas as pd
import os
from sqlalchemy import create_engine
from functools import wraps

app = Flask(__name__)
app.secret_key = 'super_secret_key_123'  # Change this in production

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

from datetime import datetime

# Load allowed project names from Excel
PROJECTS_TO_SHOW_PATH = os.path.join(os.path.dirname(__file__), 'projects.to.show.xlsx')
projects_to_show_df = pd.read_excel(PROJECTS_TO_SHOW_PATH)
PROJECTS_TO_SHOW = set(projects_to_show_df['PROJECT_NAME'].dropna().astype(str).str.strip())

# User credentials (username: password)
USERS = {
    'yasser': 'yasser123',
    'nada': 'nada123',
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in USERS and USERS[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        query_projects = "SELECT DISTINCT RECEIPT_PRJ_NAME FROM receipts_4_Report WHERE RECEIPT_PRJ_NAME IS NOT NULL"
        projects_df = pd.read_sql(query_projects, engine)
        project_names = projects_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        # Filter to only those in the allowed list
        filtered_project_names = sorted([p for p in project_names if p in PROJECTS_TO_SHOW])

        query_clients = "SELECT DISTINCT CUSTOMER_NAME FROM receipts_4_Report WHERE CUSTOMER_NAME IS NOT NULL"
        clients_df = pd.read_sql(query_clients, engine)
        client_names = clients_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        client_names.sort()

        return render_template('index.html', projects=filtered_project_names, clients=client_names)

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
            query += " AND RECEIPT_PRJ_NAME = %s"
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

        df = pd.read_sql(query, engine, params=tuple(params))
        if df.empty:
            print("⚠️ No data found.")
            return None

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

        summary_df = df[["RPT_Prj. Name", "Invoice No."]].drop_duplicates().copy()
        summary_df["Value Of Work"] = summary_df.apply(
            lambda row: df.loc[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) & 
                (df["Invoice No."] == row["Invoice No."]),
                "Value Of Work"
            ].max(), axis=1
        )

        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="Details", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1F4E78',
                'font_color': 'white',
                'border': 1
            })

            number_format = workbook.add_format({'num_format': '#,##0'})
            date_format = workbook.add_format({'num_format': 'DD-MMM-YY'})

            for sheet_name, data in [("Details", df), ("Summary", summary_df)]:
                worksheet = writer.sheets[sheet_name]
                max_row, max_col = data.shape

                for col_num, column_name in enumerate(data.columns):
                    worksheet.write(0, col_num, column_name, header_format)
                    worksheet.set_column(col_num, col_num, max(15, len(column_name) + 2))

                worksheet.freeze_panes(1, 0)
                table_range = f"A1:{chr(65 + max_col - 1)}{max_row + 1}"
                worksheet.add_table(table_range, {
                    'name': f'{sheet_name}_Table',
                    'columns': [{'header': col} for col in data.columns],
                    'style': 'Table Style Medium 9'
                })

        print(f"✅ Excel report generated with summary: {file_path}")
        return file_path

    except Exception as e:
        print("❌ Error generating Excel:")
        print(traceback.format_exc())
        return None

@app.route('/download', methods=['POST'])
@login_required
def download_excel():
    project_name = request.form.get("project_name", "all")
    client_name = request.form.get("client_name", "all")
    from_date = request.form.get("from_date") or None
    to_date = request.form.get("to_date") or None

    file_path = fetch_data(project_name, client_name, from_date, to_date)
    if file_path and os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)

    return "Error: Could not generate the report.", 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5005)  # Run on port 5005 for the login-protected receipts site 