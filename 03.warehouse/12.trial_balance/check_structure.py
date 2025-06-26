import mysql.connector as mysql
import pandas as pd
from mysql.connector import Error

# Database connection parameters
HOST = '10.10.11.242'
DATABASE = 'RME_TEST'
USERNAME = 'omar2'
PASSWORD = 'Omar_54321'

# Excel file path
EXCEL_FILE = r"D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\03.warehouse\12.trial_balance\SWD_Periods_Trial_Balance___De_170425.xlsx"

def get_db_structure():
    """Get the structure of the Trial_Balance table from the database"""
    try:
        # Establish connection
        conn = mysql.connect(
            host=HOST,
            user=USERNAME,
            password=PASSWORD,
            database=DATABASE
        )
        
        if not conn.is_connected():
            print("Failed to connect to database")
            return None
            
        cursor = conn.cursor(dictionary=True)
        
        # Get column information
        cursor.execute("""
            SELECT 
                COLUMN_NAME as name,
                DATA_TYPE as type,
                IS_NULLABLE as nullable
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'Trial_Balance'
            ORDER BY ORDINAL_POSITION
        """)
        
        column_info = cursor.fetchall()
        
        print("\nDatabase Table Structure:")
        print("-" * 80)
        for col in column_info:
            print(f"Column: {col['name']:<20} Type: {col['type']:<15} Nullable: {col['nullable']}")
        
        return column_info
    
    except Error as e:
        print(f"Error getting database structure: {str(e)}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def get_excel_structure():
    """Get the structure of the Excel file"""
    try:
        # Read Excel file
        print(f"\nReading Excel file: {EXCEL_FILE}")
        df = pd.read_excel(EXCEL_FILE)
        
        # Get column information
        excel_info = []
        for column in df.columns:
            dtype = str(df[column].dtype)
            excel_info.append({
                'name': column,
                'type': dtype,
                'sample_value': str(df[column].iloc[0]) if len(df) > 0 else 'N/A'
            })
        
        print("\nExcel File Structure:")
        print("-" * 80)
        for col in excel_info:
            print(f"Column: {col['name']:<20} Type: {col['type']:<15} Sample Value: {col['sample_value']}")
        
        return excel_info
    
    except Exception as e:
        print(f"Error reading Excel structure: {str(e)}")
        return None

def compare_structures(db_structure, excel_structure):
    """Compare database and Excel structures"""
    if not db_structure or not excel_structure:
        return
    
    print("\nStructure Comparison:")
    print("-" * 80)
    
    # Get column names
    db_columns = {col['name'].lower() for col in db_structure}
    excel_columns = {col['name'].lower() for col in excel_structure}
    
    # Find differences
    only_in_db = db_columns - excel_columns
    only_in_excel = excel_columns - db_columns
    common_columns = db_columns & excel_columns
    
    print("\nColumns only in database:")
    for col in only_in_db:
        print(f"- {col}")
    
    print("\nColumns only in Excel:")
    for col in only_in_excel:
        print(f"- {col}")
    
    print("\nCommon columns:")
    for col in common_columns:
        print(f"- {col}")

def main():
    print("Checking database and Excel file structures...")
    
    # Get structures and compare
    db_structure = get_db_structure()
    excel_structure = get_excel_structure()
    
    if db_structure and excel_structure:
        compare_structures(db_structure, excel_structure)

if __name__ == "__main__":
    main() 