import logging
from flask import Flask, render_template, send_file, request, jsonify
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
TABLE_NAME = "SWD_Collection_Report"

# Directory to store Excel files
OUTPUT_DIR = r"D:\site\rev07\downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from datetime import datetime

@app.route('/')
def index():
    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        # Fetch project names using PROJECT_NAME from SWD_Collection_Report
        query_projects = "SELECT DISTINCT PROJECT_NAME FROM SWD_Collection_Report WHERE PROJECT_NAME IS NOT NULL"
        projects_df = pd.read_sql(query_projects, engine)
        project_names = projects_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        project_names.sort()

        logging.info(f"Fetched {len(project_names)} project names from the database.")

        return render_template('index_swd.html', projects=project_names)

    except Exception as e:
        logging.error(f"Error fetching dropdown data: {e}")
        return render_template('index_swd.html', projects=[])


@app.route('/get_total_func_amount', methods=['POST'])
def get_total_func_amount():
    project_name = request.form.get('project_name')
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')

    logging.info(f"Received request to calculate total FUNC_AMOUNT for project: {project_name}, from {from_date} to {to_date}")

    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        query = f"""
            SELECT SUM(FUNC_AMOUNT) AS total_func_amount
            FROM {TABLE_NAME}
            WHERE PROJECT_NAME = %s
        """

        params = [project_name]
        
        # Apply date filters if provided
        if from_date:
            query += " AND RECEIPT_DATE >= %s"
            params.append(from_date)
            logging.info(f"From date filter applied: {from_date}")

        if to_date:
            query += " AND RECEIPT_DATE <= %s"
            params.append(to_date)
            logging.info(f"To date filter applied: {to_date}")

        result = pd.read_sql(query, engine, params=tuple(params))

        total_func_amount = result['total_func_amount'].iloc[0] if not result.empty else 0
        # Remove decimals (round to integer)
        total_func_amount = int(total_func_amount)

        # Format the total amount with commas for readability
        total_func_amount = "{:,}".format(total_func_amount)

        logging.info(f"Total FUNC_AMOUNT for project '{project_name}' from {from_date} to {to_date}: {total_func_amount}")
        return jsonify({'total_func_amount': total_func_amount})

    except Exception as e:
        logging.error(f"Error fetching total func amount: {e}")
        return jsonify({'total_func_amount': 0})


def fetch_data(project_name, from_date, to_date):
    try:
        db_uri = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(db_uri)

        query = f"""
            SELECT 
                CASH_RECEIPT_ID AS Cash_Receipt_ID,  
                RECEIPT_DATE AS Receipt_Date, 
                FUNC_AMOUNT AS Func_Amount, 
                COMMENTS,
                PROJECT_NAME AS Project_Name, 
                PROJECT_NUM AS Project_Num
            FROM {TABLE_NAME}
            WHERE 1=1
        """

        params = []

        if project_name != "all":
            query += " AND PROJECT_NAME = %s"
            params.append(project_name)

        if from_date:
            query += " AND RECEIPT_DATE >= %s"
            params.append(from_date)
            logging.info(f"From date filter applied: {from_date}")

        if to_date:
            query += " AND RECEIPT_DATE <= %s"
            params.append(to_date)
            logging.info(f"To date filter applied: {to_date}")

        logging.info(f"Executing query: {query} with params: {params}")
        df = pd.read_sql(query, engine, params=tuple(params))

        if df.empty:
            logging.warning(f"No data found for the selected filters.")
            return None

        # Read the mapping Excel for customer names
        mapping_path = os.path.join(os.path.dirname(__file__), "project-name-with-customer.xlsx")
        mapping_df = pd.read_excel(mapping_path)

        # Merge on project name
        df = df.merge(
            mapping_df,
            left_on="Project_Name",
            right_on="RECEIPT_PRJ_NAME",
            how="left"
        )

        # Rename CUSTOMER_NAME to Customer_Name for output consistency
        df = df.rename(columns={"CUSTOMER_NAME": "Customer_Name"})

        # Ensure the output columns are in the desired order
        output_columns = [
            "Cash_Receipt_ID",
            "Receipt_Date",
            "Func_Amount",
            "COMMENTS",
            "Project_Name",
            "Project_Num",
            "Customer_Name"
        ]
        df = df[output_columns]

        file_name = f"SWD_Collection_Report_{project_name}.xlsx".replace(" ", "_")
        file_path = os.path.join(OUTPUT_DIR, file_name)

        logging.info(f"Data fetched for project {project_name}, writing to file: {file_path}")

        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="SWD_Collection_Report", index=False)

            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1F4E78',
                'font_color': 'white',
                'border': 1
            })

            number_format = workbook.add_format({'num_format': '#,##0'})  # Format numbers with commas
            date_format = workbook.add_format({'num_format': 'DD-MMM-YY'})

            worksheet = writer.sheets["SWD_Collection_Report"]
            max_row, max_col = df.shape

            for col_num, column_name in enumerate(df.columns):
                worksheet.write(0, col_num, column_name, header_format)
                worksheet.set_column(col_num, col_num, max(15, len(column_name) + 2))

            worksheet.freeze_panes(1, 0)
            table_range = f"A1:{chr(65 + max_col - 1)}{max_row + 1}"
            worksheet.add_table(table_range, {
                'name': f'SWD_Collection_Report_Table',
                'columns': [{'header': col} for col in df.columns],
                'style': 'Table Style Medium 9'
            })

        logging.info(f"✅ Excel report generated: {file_path}")
        return file_path

    except Exception as e:
        logging.error(f"❌ Error generating Excel: {e}")
        return None


@app.route('/download', methods=['POST'])
def download_excel():
    project_name = request.form.get("project_name", "all")
    from_date = request.form.get("from_date") or None
    to_date = request.form.get("to_date") or None

    file_path = fetch_data(project_name, from_date, to_date)
    if file_path and os.path.exists(file_path):
        logging.info(f"File found: {file_path}. Sending file to user.")
        return send_file(file_path, as_attachment=True)

    logging.error("No data found or file could not be generated.")
    return render_template('index_swd.html', error="No data found for the selected filters.")


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)  # Change port to 5001 for the new SWD page 