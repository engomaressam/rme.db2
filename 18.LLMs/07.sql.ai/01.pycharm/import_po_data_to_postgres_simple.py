import pandas as pd
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
try:
    load_dotenv()
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv
    load_dotenv()

# Setup logging to save terminal output
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'po_import_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

def log_message(message):
    """Log message to both console and log file"""
    print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}\n')
    except Exception as e:
        print(f"Error writing to log file: {e}")

log_message(f"Starting import process. Log file: {log_file}")

# Install required packages if not available
try:
    import openpyxl
except ImportError:
    log_message("Installing missing dependency: openpyxl")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    log_message("openpyxl installed successfully.")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, types
except ImportError:
    log_message("Installing missing dependency: sqlalchemy")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
    log_message("sqlalchemy installed successfully.")
    from sqlalchemy import create_engine, text, types

try:
    import psycopg2
except ImportError:
    log_message("Installing missing dependency: psycopg2-binary")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
        import psycopg2
        log_message("psycopg2-binary installed successfully.")
    except Exception as e:
        log_message(f"Error installing psycopg2-binary: {e}")
        log_message("\nPlease install it manually with: pip install psycopg2-binary")
        log_message("Or if you're on Windows, you may need to install the appropriate Visual C++ build tools.")
        sys.exit(1)

# Test direct psycopg2 connection before proceeding
def test_direct_connection():
    try:
        log_message("Testing direct psycopg2 connection...")
        conn = psycopg2.connect(
            host=DB_PARAMS['host'],
            port=DB_PARAMS['port'],
            user=DB_PARAMS['user'],
            password=DB_PARAMS['password'],
            database=DB_PARAMS['database']
        )
        conn.close()
        log_message("Direct psycopg2 connection test successful!")
        return True
    except Exception as e:
        log_message(f"Direct connection test failed: {e}")
        return False

# Define the path to the Excel file
excel_path = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\07.sql.ai\01.erp.table\RME_PO_Follow_up_180625.xlsx"

# PostgreSQL connection parameters - Using environment variables for security
import os
DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public',
}

# Table name for the imported data
TABLE_NAME = "po_followup_from_erp"

# Function to create database connection
def get_db_connection():
    try:
        # Create connection string with URL-encoded username and password
        import urllib.parse
        user = urllib.parse.quote_plus(DB_PARAMS['user'])
        password = urllib.parse.quote_plus(DB_PARAMS['password'])
        host = DB_PARAMS['host']
        port = DB_PARAMS['port']
        database = DB_PARAMS['database']
        conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # Print connection string with password masked for debugging
        masked_conn = conn_string.replace(password, '****')
        log_message(f"Connecting with: {masked_conn}")
        log_message(f"DEBUG: Actual connection string: {conn_string}")

        # Create engine and test connection
        engine = create_engine(conn_string)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        log_message("Database connection successful!")
        return engine
    except Exception as e:
        log_message(f"Error connecting to database: {e}")
        log_message("Please verify your PostgreSQL credentials and server status.")
        log_message("Make sure host and user are specified correctly and separately.")
        sys.exit(1)

# Check if the file exists
if not os.path.exists(excel_path):
    log_message(f"Error: The file {excel_path} does not exist.")
    sys.exit(1)

# Main function to process Excel file and import to PostgreSQL
def import_data_to_postgres():
    try:
        # First test direct connection before proceeding
        if not test_direct_connection():
            log_message("WARNING: Direct connection test failed, but trying to proceed with SQLAlchemy connection anyway.")

        log_message("Reading Excel file...")
        df = pd.read_excel(excel_path)
        log_message(f"Successfully read Excel file with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Clean column names - replace spaces with underscores and convert to lowercase
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '').replace('(', '').replace(')', '') for col in df.columns]

        log_message("\nCleaned column names:")
        for i, col in enumerate(df.columns):
            log_message(f"{i+1}. {col}")

        # Define date columns
        date_columns = ['creation_date', 'approved_date']

        # Convert date columns to proper datetime format
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                log_message(f"Converted {col} to datetime")

        # Replace NaN values with None for proper SQL NULL handling
        df = df.replace({np.nan: None})

        # Define data types for SQLAlchemy
        dtype_mapping = {}
        for col in df.columns:
            if col in date_columns:
                dtype_mapping[col] = types.DateTime()
            elif df[col].dtype == 'int64':
                dtype_mapping[col] = types.BigInteger()
            elif df[col].dtype == 'float64':
                dtype_mapping[col] = types.Float()
            elif col in ['description', 'po_comments']:
                # Use Text type for potentially long text fields
                dtype_mapping[col] = types.Text()
            else:
                dtype_mapping[col] = types.String(length=255)

        # Create database connection
        engine = get_db_connection()
        schema_name = DB_PARAMS['schema']

        log_message(f"Using schema: {schema_name} and table: {TABLE_NAME}")

        # First drop the table if it exists
        with engine.connect() as connection:
            try:
                connection.execute(text(f"DROP TABLE IF EXISTS {schema_name}.{TABLE_NAME}"))
                connection.commit()
                log_message(f"Dropped existing table {schema_name}.{TABLE_NAME}")
            except Exception as e:
                log_message(f"Warning when dropping table: {e}")
                connection.rollback()

        # Then create the table and import data
        log_message(f"Creating new table with {len(df.columns)} columns and importing {len(df)} rows...")
        df.to_sql(
            name=TABLE_NAME,
            con=engine,
            schema=schema_name,
            if_exists='replace',
            index=False,
            dtype=dtype_mapping,
            chunksize=1000  # Import in chunks to avoid memory issues
        )

        # Add a primary key and create indexes
        with engine.connect() as connection:
            try:
                connection.execute(text(f"ALTER TABLE {schema_name}.{TABLE_NAME} ADD COLUMN id SERIAL PRIMARY KEY"))
                connection.commit()
                log_message("Added SERIAL PRIMARY KEY column 'id'")

                # Create indexes on common query columns
                for col in ['po_num', 'vendor', 'creation_date']:
                    if col in df.columns:
                        connection.execute(text(f"CREATE INDEX idx_{TABLE_NAME}_{col.replace('.', '_')} ON {schema_name}.{TABLE_NAME} ({col})"))
                        connection.commit()
                        log_message(f"Created index on column '{col}'")
            except Exception as e:
                log_message(f"Error when adding primary key or indexes: {e}")
                connection.rollback()

        # Print table summary
        with engine.connect() as connection:
            # Get row count
            result = connection.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{TABLE_NAME}"))
            row_count = result.scalar()
            log_message(f"\nTotal rows in table: {row_count}")

            # Sample query to show data was imported correctly
            log_message("\nSample data from imported table (first 3 rows):")
            result = connection.execute(text(f"""
                SELECT po_num, vendor, description, amount_egp, creation_date 
                FROM {schema_name}.{TABLE_NAME} 
                LIMIT 3
            """))
            columns = result.keys()
            rows = result.fetchall()

            # Print column headers
            header = "| " + " | ".join(str(col) for col in columns) + " |"
            log_message(header)
            log_message("-" * len(header))

            # Print rows
            for row in rows:
                log_message("| " + " | ".join(str(val)[:20] + ("..." if val and len(str(val)) > 20 else "") for val in row) + " |")

        log_message(f"\nSuccessfully imported {df.shape[0]} rows to table '{TABLE_NAME}' in schema '{schema_name}'.")
        log_message("\nSuggested SQL for rev.19 AI:")
        log_message(f"""
        SELECT 
            po_num, project_name, vendor, description, 
            quantity, unit_price, amount_egp, creation_date, 
            po_status, qty_received, qty_delivered
        FROM {schema_name}.{TABLE_NAME}
        WHERE creation_date >= '2024-01-01'
        ORDER BY creation_date DESC
        LIMIT 100;
        """)

        log_message("\nImport process completed successfully.")

    except Exception as e:
        error_msg = f"Error during import process: {e}"
        log_message(error_msg)
        log_message(traceback.format_exc())
        log_message(f"Check the log file for complete error details: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        import_data_to_postgres()
        log_message(f"\nImport completed. Full log saved to: {log_file}")
    except Exception as e:
        log_message(f"Unhandled exception: {e}")
        log_message(traceback.format_exc())
        log_message(f"Check the log file for complete error details: {log_file}")
        sys.exit(1)
