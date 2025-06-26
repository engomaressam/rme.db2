import pandas as pd
import numpy as np
import os
import sys
import subprocess

# Install required packages if not available
try:
    import openpyxl
except ImportError:
    print("Installing missing dependency: openpyxl")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl installed successfully.")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text, types
except ImportError:
    print("Installing missing dependency: sqlalchemy")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
    print("sqlalchemy installed successfully.")
    from sqlalchemy import create_engine, text, types

try:
    import psycopg2
except ImportError:
    print("Installing missing dependency: psycopg2")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    print("psycopg2 installed successfully.")

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

# Get database connection parameters from Django settings
db_settings = settings.DATABASES['default']

# Define the path to the Excel file
excel_path = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\07.sql.ai\01.erp.table\RME_PO_Follow_up_180625.xlsx"

# PostgreSQL connection parameters - MODIFY THESE TO MATCH YOUR DATABASE
DB_PARAMS = {
    'host': 'localhost',     # Your PostgreSQL server address
    'port': '5432',         # Default PostgreSQL port
    'user': 'postgres',     # Your PostgreSQL username
    'password': 'admin',    # Your PostgreSQL password
    'database': 'postgres', # Your PostgreSQL database name
    'schema': 'public'      # Schema where you want to create the table
}

# Table name for the imported data
TABLE_NAME = "po_followup_from_erp"

# Function to create database connection
def get_db_connection():
    try:
        # Create connection string
        conn_string = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
        engine = create_engine(conn_string)
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

# Check if the file exists
if not os.path.exists(excel_path):
    print(f"Error: The file {excel_path} does not exist.")
    sys.exit(1)

# Read the Excel file
try:
    print("Reading Excel file...")
    df = pd.read_excel(excel_path)
    print(f"Successfully read Excel file with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Clean column names - replace spaces with underscores and convert to lowercase
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_').replace('.', '').replace('(', '').replace(')', '') for col in df.columns]

    print("\nCleaned column names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    # Define column data types based on content
    date_columns = ['creation_date', 'approved_date']
    numeric_columns = [
        'po_num', 'pr_num', 'vendor_no', 'po_line', 'pr_line', 'quantity', 'qty_cancelled', 'unit_price', 
        'unit_price_without_tax', 'amount', 'amount_egp', 'amount_without_tax', 'amount_egp_without_tax', 
        'tax_amountegp', 'tax_amount', 'qty_received', 'qty_accepted', 'qty_rejected', 'qty_delivered',
        'quantity_open_amount', 'docs'
    ]

    text_columns = [
        'cost_center_number', 'project_name', 'organization_code', 'pr_category', 'po_status', 'shipment_cancel_status', 
        'shipment_close_status', 'next_approver', 'vendor', 'buyer', 'buyer_dept', 'pr_reason', 'po_comments', 
        'store_code', 'description', 'uom', 'currency', 'tax_code', 'task', 'task_name', 'expenditure_type', 
        'expenditure_category', 'term', 'qty_open'
    ]

    # Validate and handle data types
    print("\nProcessing columns by data type...")

    # Handle date columns
    for col in date_columns:
        if col in df.columns:
            print(f"Converting {col} to datetime")
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Handle numeric columns - ensure proper type conversion
    for col in numeric_columns:
        if col in df.columns:
            original_type = df[col].dtype
            try:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} from {original_type} to {df[col].dtype}")
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")

    # Handle text columns - ensure they're strings
    for col in text_columns:
        if col in df.columns:
            # Convert to string but handle None/NaN values appropriately
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

    # Replace NaN values with None for proper SQL NULL handling
    df = df.replace({np.nan: None})

    # Report null values percentage per column
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    print("\nNull values percentage by column:")
    for col, percent in null_percent.items():
        if percent > 5:  # Only show columns with significant null percentages
            print(f"{col}: {percent:.2f}%")

    # Show sample data
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())

    # Create database connection
    engine, db_settings = get_db_connection()

    # Define data types for SQLAlchemy
    dtype_mapping = {}
    for col in df.columns:
        if col in date_columns:
            dtype_mapping[col] = types.DateTime()
        elif df[col].dtype == 'int64':
            dtype_mapping[col] = types.BigInteger()
        elif df[col].dtype == 'float64':
            dtype_mapping[col] = types.Float()
        elif col == 'description' or col == 'po_comments':
            # Use Text type for potentially long text fields
            dtype_mapping[col] = types.Text()
        else:
            # Calculate maximum length for string columns
            max_len = df[col].astype(str).str.len().max()
            if pd.isna(max_len) or max_len < 50:
                dtype_mapping[col] = types.String(length=100)  # Default for short strings
            elif max_len < 255:
                dtype_mapping[col] = types.String(length=255)  # Medium strings
            else:
                dtype_mapping[col] = types.Text()  # For long text

    # Create table and import data
    print("Importing data to PostgreSQL...")
    schema_name = DB_PARAMS['schema']

    print(f"Using schema: {schema_name} and table: {TABLE_NAME}")

    # First drop the table if it exists
    with engine.connect() as connection:
        try:
            connection.execute(f"DROP TABLE IF EXISTS {schema_name}.{TABLE_NAME}")
            connection.commit()
            print(f"Dropped existing table {schema_name}.{TABLE_NAME}")
        except Exception as e:
            print(f"Warning when dropping table: {e}")
            connection.rollback()

    # Then create the table and import data
    print(f"Creating new table with {len(df.columns)} columns and importing {len(df)} rows...")
    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        schema=schema_name,
        if_exists='replace',
        index=False,
        dtype=dtype_mapping,
        chunksize=500  # Import in chunks to avoid memory issues
    )

    # Add a primary key
    with engine.connect() as connection:
        try:
            connection.execute(f"ALTER TABLE {schema_name}.{TABLE_NAME} ADD COLUMN id SERIAL PRIMARY KEY")
            connection.commit()
            print("Added SERIAL PRIMARY KEY column 'id'")

            # Create indexes on common query columns
            for col in ['po_num', 'vendor', 'creation_date']:
                if col in df.columns:
                    connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_{col.replace('.', '_')} ON {schema_name}.{TABLE_NAME} ({col})")
                    connection.commit()
                    print(f"Created index on column '{col}'")
        except Exception as e:
            print(f"Error when adding primary key or indexes: {e}")
            connection.rollback()

    # Print table summary and column information
    with engine.connect() as connection:
        # Get column information
        result = connection.execute(f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{TABLE_NAME}'
            AND table_schema = '{schema_name}'
            ORDER BY ordinal_position;
        """)

        print(f"\nTable structure for {schema_name}.{TABLE_NAME}:")
        print("{:<30} {:<20} {:<15}".format("Column Name", "Data Type", "Max Length"))
        print("-" * 65)
        for row in result:
            max_len = row[2] if row[2] is not None else ''
            print("{:<30} {:<20} {:<15}".format(row[0], row[1], max_len))

        # Get row count
        result = connection.execute(f"SELECT COUNT(*) FROM {schema_name}.{TABLE_NAME}")
        row_count = result.scalar()
        print(f"\nTotal rows in table: {row_count}")

        # Sample query to show data was imported correctly
        print("\nSample data from imported table (first 3 rows, 5 columns):")
        result = connection.execute(f"""
            SELECT po_num, vendor, description, amount_egp, creation_date 
            FROM {schema_name}.{TABLE_NAME} 
            LIMIT 3
        """)
        columns = result.keys()
        rows = result.fetchall()

        # Print column headers
        print("| " + " | ".join(str(col) for col in columns) + " |")
        print("| " + " | ".join("---" for _ in columns) + " |")

        # Print rows
        for row in rows:
            print("| " + " | ".join(str(val)[:20] + ("..." if val and len(str(val)) > 20 else "") for val in row) + " |")

    print(f"\nSuccessfully imported {df.shape[0]} rows to table '{TABLE_NAME}' in schema '{schema_name}'.")
    print("This table can now be used for your rev.19 AI implementation.")
    print("\nSuggested SQL to use this data:")
    print(f"""
    SELECT 
        po_num, vendor, description, 
        quantity, unit_price, amount_egp, 
        creation_date, po_status,
        qty_received, qty_delivered
    FROM {schema_name}.{TABLE_NAME}
    WHERE creation_date >= '2024-01-01'
    ORDER BY creation_date DESC
    LIMIT 100;
    """)

except Exception as e:
    print(f"Error during import process: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

print("\nImport process completed successfully.")
