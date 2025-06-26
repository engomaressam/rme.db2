import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from datetime import datetime

# Install required packages if not available
try:
    import openpyxl
except ImportError:
    print("Installing missing dependency: openpyxl")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    print("openpyxl installed successfully.")

try:
    import matplotlib
    import seaborn
except ImportError:
    print("Installing data visualization dependencies")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    print("Data visualization dependencies installed successfully.")

# Initialize Django to get database settings
try:
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    django.setup()
    from django.conf import settings
except ImportError as e:
    print(f"Error importing Django settings: {e}")
    print("Make sure to run this script from the Django project root directory.")
    sys.exit(1)

# Database connection function
def get_db_connection():
    try:
        db_settings = settings.DATABASES['default']
        conn_string = f"postgresql://{db_settings['USER']}:{db_settings['PASSWORD']}@{db_settings['HOST']}:{db_settings['PORT']}/{db_settings['NAME']}"
        engine = create_engine(conn_string)
        return engine, db_settings
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def analyze_po_data():
    print("Analyzing PO data for use in rev.19 AI...")
    engine, db_settings = get_db_connection()

    # Get schema name from Django settings
    try:
        schema_name = db_settings.get('OPTIONS', {}).get('options', '-c search_path=public').split('=')[1]
    except (KeyError, IndexError, AttributeError):
        schema_name = 'public'

    TABLE_NAME = "po_followup_from_erp"

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
            return

    # Query the database to get useful statistics
    print("\nQuerying database for statistics...")

    # Load data into pandas for analysis
    query = f"SELECT * FROM {schema_name}.{TABLE_NAME} LIMIT 100000"  # Limit to prevent memory issues
    df = pd.read_sql(query, engine)

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

        # Show distribution by year and month
        if pd.notna(min_date) and pd.notna(max_date):
            df['year_month'] = df['creation_date'].dt.strftime('%Y-%m')
            monthly_counts = df['year_month'].value_counts().sort_index()
            print("\nPO distribution by month:")
            for month, count in monthly_counts.items():
                print(f"{month}: {count} POs")

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

    # 8. Recommendations for AI model features
    print("\nRecommended columns for rev.19 AI model:")
    recommended_cols = [
        'po_num', 'project_name', 'vendor', 'description', 'quantity', 
        'amount_egp', 'creation_date', 'po_status', 'qty_received',
        'qty_delivered'
    ]

    present_recommended = [col for col in recommended_cols if col in df.columns]
    print(", ".join(present_recommended))

    print("\nAnalysis complete. You can now use this information for your rev.19 AI implementation.")

if __name__ == "__main__":
    analyze_po_data()
