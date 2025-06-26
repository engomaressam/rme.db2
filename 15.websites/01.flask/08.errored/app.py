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

db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
engine = create_engine(db_uri)

# Directory to store Excel files
OUTPUT_DIR = r"D:\site\rev07\downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Route to load dropdown lists for Projects and Clients

from datetime import datetime 

@app.route('/')
def index():
    try:

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
        query = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
        params = []

        if project_name != "all":
            query += " AND `RPT_Prj. Name` = %s"
            params.append(project_name)

        if client_name != "all":
            query += " AND `Client Name` = %s"
            params.append(client_name)

        if from_date:
            query += " AND `RPT_Date` >= %s"
            params.append(from_date)

        if to_date:
            query += " AND `RPT_Date` <= %s"
            params.append(to_date)

        print(f"🔍 Executing SQL with params: {params}")
        df = pd.read_sql(query, engine, params=tuple(params))

        if df.empty:
            raise ValueError("No data found for the selected filters.")

        # ✅ Summary generation
        summary_data = df[["RPT_Prj. Name", "Invoice No."]].drop_duplicates()

        summary_data["Value Of Work"] = summary_data.apply(
            lambda row: df[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."])
            ]["Value Of Work"].max(), axis=1
        )

        summary_data["Net"] = summary_data.apply(
            lambda row: df[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."])
            ]["Net"].max(), axis=1
        )

        summary_data["Amount Applied"] = summary_data.apply(
            lambda row: df[
                (df["RPT_Prj. Name"] == row["RPT_Prj. Name"]) &
                (df["Invoice No."] == row["Invoice No."])
            ]["Amount Applied"].sum(), axis=1
        )

        # ✅ Create Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prj = project_name.replace(" ", "_") if project_name != "all" else "all"
        safe_clt = client_name.replace(" ", "_") if client_name != "all" else "all"
        filename = f"Receipts_Report_{safe_prj}_{safe_clt}.xlsx"
        file_path = os.path.join(OUTPUT_DIR, filename)

        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Details', index=False)
        summary_data.to_excel(writer, sheet_name='Summary', index=False)

        workbook = writer.book
        details_ws = writer.sheets['Details']
        summary_ws = writer.sheets['Summary']

        # ✅ Format settings
        number_format = workbook.add_format({'num_format': '#,##0'})
        date_format = workbook.add_format({'num_format': 'DD-MMM-YY'})
        text_format = workbook.add_format({'num_format': '@'})

        # ✅ Format Details sheet
        for col_num, col in enumerate(df.columns):
            col_width = max(12, len(col) + 2)
            col_type = df[col].dtype

            if col == "RPT_Date":
                details_ws.set_column(col_num, col_num, col_width, date_format)
            elif pd.api.types.is_numeric_dtype(col_type) and col != "RPT_ID":
                details_ws.set_column(col_num, col_num, col_width, number_format)
            else:
                details_ws.set_column(col_num, col_num, col_width, text_format)

        # ✅ Format Summary sheet
        for col_num, col in enumerate(summary_data.columns):
            col_width = max(12, len(col) + 2)
            col_type = summary_data[col].dtype

            if pd.api.types.is_numeric_dtype(col_type):
                summary_ws.set_column(col_num, col_num, col_width, number_format)
            else:
                summary_ws.set_column(col_num, col_num, col_width, text_format)

        # ✅ Freeze headers
        details_ws.freeze_panes(1, 0)
        summary_ws.freeze_panes(1, 0)

        workbook.close()
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
