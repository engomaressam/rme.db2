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

def get_excel_totals():
    """Get totals from Excel file for Jan-Apr"""
    try:
        print("\nReading Excel file...")
        df = pd.read_excel(EXCEL_FILE)
        
        # Calculate totals for each month
        totals = {}
        for month in ['Jan', 'Feb', 'Mar', 'Apr']:
            total = pd.to_numeric(df[month], errors='coerce').sum()
            totals[month] = total
            print(f"Excel {month} total: {total:,.2f}")
        
        return totals
    
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

def get_mysql_totals():
    """Get totals from MySQL database for Jan-Apr"""
    try:
        # Connect to database
        conn = mysql.connect(
            host=HOST,
            user=USERNAME,
            password=PASSWORD,
            database=DATABASE
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Calculate totals for each month
        totals = {}
        for month in ['Jan', 'Feb', 'Mar', 'Apr']:
            cursor.execute(f"""
                SELECT SUM(CAST({month} AS DECIMAL(20,2))) as total
                FROM Trial_Balance
            """)
            result = cursor.fetchone()
            total = float(result['total']) if result['total'] else 0
            totals[month] = total
            print(f"MySQL {month} total: {total:,.2f}")
        
        return totals
    
    except Error as e:
        print(f"Error querying database: {str(e)}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def compare_totals(excel_totals, mysql_totals):
    """Compare totals between Excel and MySQL"""
    if not excel_totals or not mysql_totals:
        print("\nCannot compare totals due to errors")
        return
    
    print("\nComparing Monthly Totals:")
    print("-" * 85)
    print(f"{'Month':<6} {'Excel Total':>20} {'MySQL Total':>20} {'Difference':>20} {'Status':<10}")
    print("-" * 85)
    
    # Consider a difference of less than 10 as negligible
    THRESHOLD = 10
    
    all_acceptable = True
    for month in ['Jan', 'Feb', 'Mar', 'Apr']:
        excel_val = excel_totals[month]
        mysql_val = mysql_totals[month]
        difference = abs(excel_val - mysql_val)
        
        # Check if difference is acceptable
        is_acceptable = difference <= THRESHOLD
        
        status = '✓' if is_acceptable else '❌'
        if difference < 0.01:  # For very small differences, show as 0
            difference = 0
            
        print(f"{month:<6} {excel_val:>20,.2f} {mysql_val:>20,.2f} {difference:>20,.2f} {status}")
        
        if not is_acceptable:
            all_acceptable = False
    
    print("\nVerification Result:")
    if all_acceptable:
        print("✅ All differences are less than 10!")
        print("   The data import was successful.")
    else:
        print("❌ Some differences exceed 10. Please check the values above.")

def main():
    print("Verifying totals between Excel and MySQL...")
    
    # Get totals from both sources
    excel_totals = get_excel_totals()
    mysql_totals = get_mysql_totals()
    
    # Compare the totals
    compare_totals(excel_totals, mysql_totals)

if __name__ == "__main__":
    main() 