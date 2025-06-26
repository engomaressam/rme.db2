from flask import Flask, send_file
import mysql.connector as mysql
import pandas as pd
import os

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
OUTPUT_DIR = "downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data():
    """Fetches data from MySQL and saves it as an Excel file."""
    try:
        cnx = mysql.connect(**DB_CONFIG)
        query = f"SELECT * FROM {TABLE_NAME}"
        df = pd.read_sql(query, cnx)
        
        file_path = os.path.join(OUTPUT_DIR, f"{TABLE_NAME}.xlsx")
        df.to_excel(file_path, index=False)
        cnx.close()
        return file_path
    except Exception as e:
        return str(e)


@app.route('/')
def home():
    return "<h2>Welcome to the Receipts Report Downloader</h2><p><a href='/download'>Click here to download the latest Receipts Report</a></p>"


@app.route('/download')
def download_excel():
    file_path = fetch_data()
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Error: Could not generate the report."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
