import mysql.connector as mysql
import pandas as pd
import numpy as np
from mysql.connector import Error

# Database connection parameters
HOST = '10.10.11.242'
DATABASE = 'RME_TEST'
USERNAME = 'omar2'
PASSWORD = 'Omar_54321'

# Excel file path
EXCEL_FILE = r"D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\03.warehouse\12.trial_balance\SWD_Periods_Trial_Balance___De_170425.xlsx"

def connect_to_database():
    """Establish connection to the database"""
    try:
        conn = mysql.connect(
            host=HOST,
            user=USERNAME,
            password=PASSWORD,
            database=DATABASE
        )
        print("Successfully connected to the database!")
        return conn
    except Error as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def modify_table_structure(conn):
    """Remove extra columns from Trial_Balance table"""
    try:
        cursor = conn.cursor()
        
        # Check if columns exist and drop them
        columns_to_drop = ['Year', 'Level', 'CLARIFICATION']
        
        for column in columns_to_drop:
            try:
                print(f"Attempting to drop column {column}...")
                cursor.execute(f"ALTER TABLE Trial_Balance DROP COLUMN {column}")
                conn.commit()
                print(f"Successfully dropped column {column}")
            except Error as e:
                if "check that column/key exists" in str(e):
                    print(f"Column {column} does not exist")
                else:
                    raise e
        
        print("Table structure modification completed")
            
    except Error as e:
        print(f"Error modifying table structure: {str(e)}")
        conn.rollback()
        raise

def truncate_trial_balance(conn):
    """Delete all records from Trial_Balance table"""
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Trial_Balance")
        conn.commit()
        print("Successfully deleted existing data from Trial_Balance table")
    except Error as e:
        print(f"Error deleting data: {str(e)}")
        conn.rollback()
        raise

def clean_data_for_import(df):
    """Clean and prepare data for import"""
    # Replace NaN with None
    df = df.replace({np.nan: None})
    
    # Convert numeric columns to proper format
    numeric_columns = ['Open Balance', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Adj', 'Total']
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to float first to handle any string representations
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace NaN with None after conversion
            df[col] = df[col].replace({np.nan: None})
    
    return df

def import_excel_data(conn):
    """Import data from Excel file to Trial_Balance table"""
    try:
        # Read Excel file
        print(f"Reading Excel file: {EXCEL_FILE}")
        df = pd.read_excel(EXCEL_FILE)
        
        # Clean data
        df = clean_data_for_import(df)
        
        print("\nExcel data preview:")
        print(df.head())
        print(f"\nTotal rows to import: {len(df)}")

        # Prepare data for import
        cursor = conn.cursor()
        
        # Get the column names from the dataframe
        columns = df.columns.tolist()
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join([f'`{col}`' for col in columns])
        
        # Insert data row by row
        sql = f"""
        INSERT INTO Trial_Balance (
            {columns_str}
        ) VALUES ({placeholders})
        """
        
        # Convert DataFrame to list of tuples for batch insert
        values = df.values.tolist()
        
        # Insert in batches of 1000
        batch_size = 1000
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            cursor.executemany(sql, batch)
            conn.commit()
            print(f"Imported rows {i + 1} to {min(i + batch_size, len(values))}...")
        
        print(f"\nSuccessfully imported {len(df)} rows into Trial_Balance table")
        
    except Error as e:
        print(f"Error importing data: {str(e)}")
        conn.rollback()
        raise

def main():
    # Connect to the database
    conn = connect_to_database()
    if conn is None:
        return
    
    try:
        # First modify table structure
        modify_table_structure(conn)
        
        # Delete existing data
        truncate_trial_balance(conn)
        
        # Import new data
        import_excel_data(conn)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if conn.is_connected():
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 