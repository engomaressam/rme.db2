import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import create_engine, text
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
import traceback

# Install required packages if not available
try:
    import sqlalchemy
except ImportError:
    print("Installing missing dependency: sqlalchemy")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
    print("sqlalchemy installed successfully.")

try:
    import matplotlib
except ImportError:
    print("Installing data visualization dependencies")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    print("matplotlib installed successfully.")

# PostgreSQL connection parameters - MODIFY THESE TO MATCH YOUR DATABASE
DB_PARAMS = {
    'host': 'localhost',     # Your PostgreSQL server address
    'port': '5432',         # Default PostgreSQL port
    'user': 'postgres',     # Your PostgreSQL username
    'password': 'PMO@1234', # Your PostgreSQL password
    'database': 'postgres', # Your PostgreSQL database name
    'schema': 'public'      # Schema where the table was created
}

# Setup logging to save terminal output
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'po_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

def log_message(message):
    """Log message to both console and log file"""
    print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}\n')
    except Exception as e:
        print(f"Error writing to log file: {e}")

log_message(f"Starting analysis. Log file: {log_file}")

# Install required packages if not available
try:
    import sqlalchemy
except ImportError:
    log_message("Installing missing dependency: sqlalchemy")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
    log_message("sqlalchemy installed successfully.")

try:
    import matplotlib
except ImportError:
    log_message("Installing data visualization dependencies")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    log_message("matplotlib installed successfully.")

# PostgreSQL connection parameters - MODIFY THESE TO MATCH YOUR DATABASE
DB_PARAMS = {
    'host': 'localhost',     # Your PostgreSQL server address
    'port': '5432',         # Default PostgreSQL port
    'user': 'postgres',     # Your PostgreSQL username
    'password': 'PMO@1234',    # Your PostgreSQL password
    'database': 'postgres', # Your PostgreSQL database name
    'schema': 'public'      # Schema where the table was created
}

TABLE_NAME = "po_followup_from_erp"

# Function to create database connection
def get_db_connection():
    try:
        import urllib.parse
        user = urllib.parse.quote_plus(DB_PARAMS['user'])
        password = urllib.parse.quote_plus(DB_PARAMS['password'])
        host = DB_PARAMS['host']
        port = DB_PARAMS['port']
        database = DB_PARAMS['database']
        conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        log_message(f"DEBUG: Actual connection string: {conn_string}")
        engine = create_engine(conn_string)
        return engine
    except Exception as e:
        log_message(f"Error connecting to database: {e}")
        sys.exit(1)

def analyze_po_data():
    log_message("Analyzing PO data for use in rev.19 AI...")
    engine = get_db_connection()
    schema_name = DB_PARAMS['schema']

    # Check if table exists
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema_name}' 
                AND table_name = '{TABLE_NAME}'
            );
        """))
        table_exists = result.scalar()

        if not table_exists:
            log_message(f"Error: Table {schema_name}.{TABLE_NAME} does not exist.")
            log_message("Please run import_po_data_to_postgres.py first.")
            return

    # Query the database to get useful statistics
    log_message("\nQuerying database for statistics...")

    # Load data into pandas for analysis
    query = f"SELECT * FROM {schema_name}.{TABLE_NAME} LIMIT 50000"  # Limit to prevent memory issues
    df = pd.read_sql(text(query), engine)

    # 1. General statistics
    log_message(f"\nDataset size: {len(df)} rows and {len(df.columns)} columns")

    # 2. Important columns for AI analysis
    key_columns = [
        'po_num', 'project_name', 'vendor', 'description', 'quantity', 'unit_price',
        'amount', 'amount_egp', 'creation_date', 'po_status', 'qty_received',
        'qty_delivered', 'qty_open'
    ]

    present_columns = [col for col in key_columns if col in df.columns]
    log_message(f"\nKey columns for AI analysis: {present_columns}")

    # 3. Data completeness for key columns
    log_message("\nData completeness for key columns:")
    for col in present_columns:
        null_percent = df[col].isna().mean() * 100
        log_message(f"{col}: {100-null_percent:.2f}% complete")

    # 4. Date range analysis
    if 'creation_date' in df.columns:
        log_message("\nDate range analysis:")
        min_date = df['creation_date'].min()
        max_date = df['creation_date'].max()
        log_message(f"PO data spans from {min_date} to {max_date}")

    # 5. Vendor analysis
    if 'vendor' in df.columns:
        vendor_counts = df['vendor'].value_counts().head(10)
        log_message("\nTop 10 vendors by PO count:")
        for vendor, count in vendor_counts.items():
            log_message(f"{vendor}: {count} POs")

    # 6. Status analysis
    if 'po_status' in df.columns:
        status_counts = df['po_status'].value_counts()
        log_message("\nPO status distribution:")
        for status, count in status_counts.items():
            log_message(f"{status}: {count} POs ({count/len(df)*100:.2f}%)")

    # 7. Project analysis
    if 'project_name' in df.columns:
        project_counts = df['project_name'].value_counts().head(10)
        log_message("\nTop 10 projects by PO count:")
        for project, count in project_counts.items():
            log_message(f"{project}: {count} POs")

    # 8. Recommendations for rev.19 AI model columns
    log_message("\nRecommended columns for rev.19 AI model:")
    log_message("""po_num, project_name, vendor, description, quantity, 
amount_egp, creation_date, po_status, qty_received, qty_delivered""")

    # 9. Example SQL query for rev.19 AI
    log_message("\nExample SQL query for rev.19 AI:")
    log_message(f"""
    SELECT 
        po_num, project_name, vendor, description, quantity,
        unit_price, amount_egp, creation_date, po_status, 
        qty_received, qty_delivered
    FROM {schema_name}.{TABLE_NAME}
    WHERE creation_date >= '2024-01-01'
    ORDER BY creation_date DESC
    LIMIT 100;
    """)

    log_message("\nAnalysis complete. You can now use this information for your rev.19 AI implementation.")
    log_message(f"Full analysis log saved to: {log_file}")

if __name__ == "__main__":
    try:
        analyze_po_data()
    except Exception as e:
        log_message(f"Error during analysis: {e}")
        log_message(traceback.format_exc())
        log_message(f"Check the log file for complete error details: {log_file}")
        sys.exit(1)

    # Check if table exists
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema_name}' 
                AND table_name = '{TABLE_NAME}'
            );
        """))
        table_exists = result.scalar()

        if not table_exists:
            print(f"Error: Table {schema_name}.{TABLE_NAME} does not exist.")
            print("Please run import_po_data_to_postgres.py first.")
            sys.exit(1)

    # Query the database to get useful statistics
    print("\nQuerying database for statistics...")

    # Load data into pandas for analysis
    query = f"SELECT * FROM {schema_name}.{TABLE_NAME} LIMIT 50000"  # Limit to prevent memory issues
    df = pd.read_sql(text(query), engine)

    # 1. General statistics
    print(f"\nDataset size: {len(df)} rows and {len(df.columns)} columns")

    # 2. Important columns for AI analysis
    key_columns = [
        'po_num', 'project_name', 'vendor', 'description', 'quantity', 'unit_price',
        'amount', 'amount_egp', 'creation_date', 'po_status', 'qty_received',
        'qty_delivered', 'qty_open'
    ]

    present_columns = [col for col in key_columns if col in df.columns]
    print(f"\nKey columns for AI analysis: {present_columns}")

    # 3. Data completeness for key columns
    print("\nData completeness for key columns:")
    for col in present_columns:
        null_percent = df[col].isna().mean() * 100
        print(f"{col}: {100-null_percent:.2f}% complete")

    # 4. Date range analysis
    if 'creation_date' in df.columns:
        print("\nDate range analysis:")
        min_date = df['creation_date'].min()
        max_date = df['creation_date'].max()
        print(f"PO data spans from {min_date} to {max_date}")

    # 5. Vendor analysis
    if 'vendor' in df.columns:
        vendor_counts = df['vendor'].value_counts().head(10)
        print("\nTop 10 vendors by PO count:")
        for vendor, count in vendor_counts.items():
            print(f"{vendor}: {count} POs")

    # 6. Status analysis
    if 'po_status' in df.columns:
        status_counts = df['po_status'].value_counts()
        print("\nPO status distribution:")
        for status, count in status_counts.items():
            print(f"{status}: {count} POs ({count/len(df)*100:.2f}%)")

    # 7. Project analysis
    if 'project_name' in df.columns:
        project_counts = df['project_name'].value_counts().head(10)
        print("\nTop 10 projects by PO count:")
        for project, count in project_counts.items():
            print(f"{project}: {count} POs")

    # 8. Recommendations for rev.19 AI model columns
    print("\nRecommended columns for rev.19 AI model:")
    print("""po_num, project_name, vendor, description, quantity, 
amount_egp, creation_date, po_status, qty_received, qty_delivered""")

    # 9. Example SQL query for rev.19 AI
    print("\nExample SQL query for rev.19 AI:")
    print(f"""
    SELECT 
        po_num, project_name, vendor, description, quantity,
        unit_price, amount_egp, creation_date, po_status, 
        qty_received, qty_delivered
    FROM {schema_name}.{TABLE_NAME}
    WHERE creation_date >= '2024-01-01'
    ORDER BY creation_date DESC
    LIMIT 100;
    """)

    print("\nAnalysis complete. You can now use this information for your rev.19 AI implementation.")

if __name__ == "__main__":
    analyze_po_data()
