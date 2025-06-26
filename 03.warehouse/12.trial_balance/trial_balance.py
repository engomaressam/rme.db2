import pyodbc
import pandas as pd
from datetime import datetime

# Database connection parameters
SERVER = '10.10.11.242'
DATABASE = 'RME_TEST'
USERNAME = 'omar2'
PASSWORD = 'Omar_54321'

# Create connection string
conn_str = (
    f'DRIVER={{SQL Server}};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    f'UID={USERNAME};'
    f'PWD={PASSWORD};'
)

def connect_to_database():
    """Establish connection to the database"""
    try:
        conn = pyodbc.connect(conn_str)
        print("Successfully connected to the database!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def get_trial_balance(conn):
    """Fetch data from Trial_Balance table"""
    query = """
    SELECT 
        Year,
        [Account Code],
        [Account Name],
        Level,
        CLARIFICATION,
        [Sub Account],
        [Value]
    FROM Trial_Balance
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

def main():
    # Connect to the database
    conn = connect_to_database()
    if conn is None:
        return
    
    try:
        # Get trial balance data
        df = get_trial_balance(conn)
        if df is not None:
            print("\nTrial Balance Data Preview:")
            print(df.head())
            
            # Save to Excel file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = f"trial_balance_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"\nData exported to {excel_file}")
            
    finally:
        conn.close()

if __name__ == "__main__":
    main() 