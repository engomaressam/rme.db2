import mysql.connector
from mysql.connector import Error
import pandas as pd
from tabulate import tabulate

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def connect_to_database():
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
        return None
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def compare_table_row_counts():
    """Compare the row counts of both tables"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    try:
        # Count rows in original table
        cursor.execute("SELECT COUNT(*) FROM RME_PO_Follow_Up_Report")
        original_count = cursor.fetchone()[0]
        
        # Count rows in merged table
        cursor.execute("SELECT COUNT(*) FROM po_followup_merged")
        merged_count = cursor.fetchone()[0]
        
        print("\n=== ROW COUNT COMPARISON ===")
        print(f"Original table (RME_PO_Follow_Up_Report): {original_count} rows")
        print(f"Merged table (po_followup_merged): {merged_count} rows")
        print(f"Difference: {original_count - merged_count} rows")
    except Error as e:
        print(f"Error comparing row counts: {e}")
    finally:
        cursor.close()
        connection.close()

def compare_column_data():
    """Compare the data in the columns that were copied from original to merged table"""
    connection = connect_to_database()
    if not connection:
        return
    
    # Get the first 5 rows from both tables for comparison
    try:
        # Get columns from original table
        query_orig = """
        SELECT id, PO_NUM, COMMENTS, APPROVED_DATE, UOM, 
               ITEM_DESCRIPTION, UNIT_PRICE, QUANTITY_RECEIVED, 
               LINE_AMOUNT, PROJECT_NAME, VENDOR_NAME 
        FROM RME_PO_Follow_Up_Report 
        LIMIT 5
        """
        df_orig = pd.read_sql(query_orig, connection)
        
        # Get same columns from merged table
        query_merged = """
        SELECT id, PO_NUM, COMMENTS, APPROVED_DATE, UOM, 
               ITEM_DESCRIPTION, UNIT_PRICE, QUANTITY_RECEIVED, 
               LINE_AMOUNT, PROJECT_NAME, VENDOR_NAME 
        FROM po_followup_merged 
        LIMIT 5
        """
        df_merged = pd.read_sql(query_merged, connection)
        
        print("\n=== SAMPLE DATA COMPARISON (First 5 rows) ===")
        print("\nOriginal Table Data:")
        print(tabulate(df_orig, headers='keys', tablefmt='psql'))
        
        print("\nMerged Table Data:")
        print(tabulate(df_merged, headers='keys', tablefmt='psql'))
    except Error as e:
        print(f"Error comparing column data: {e}")
    finally:
        connection.close()

def check_for_missing_data():
    """Check if any PO_NUMs from original table are missing in merged table"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    try:
        # Find PO_NUMs in original but not in merged
        cursor.execute("""
        SELECT PO_NUM FROM RME_PO_Follow_Up_Report
        WHERE PO_NUM NOT IN (SELECT PO_NUM FROM po_followup_merged)
        LIMIT 10
        """)
        missing_pos = cursor.fetchall()
        
        print("\n=== MISSING PO NUMBERS ===")
        if missing_pos:
            print("The following PO numbers are in the original table but missing in the merged table (up to 10):")
            for po in missing_pos:
                print(f"- {po[0]}")
        else:
            print("No missing PO numbers detected. All PO numbers from the original table exist in the merged table.")
    except Error as e:
        print(f"Error checking for missing data: {e}")
    finally:
        cursor.close()
        connection.close()

def check_terms_merging():
    """Verify that terms were properly merged for select POs"""
    connection = connect_to_database()
    if not connection:
        return
    
    try:
        # Sample some POs that have terms
        query = """
        SELECT pm.PO_NUM, pm.TERMS 
        FROM po_followup_merged pm
        WHERE pm.TERMS IS NOT NULL AND pm.TERMS != ''
        LIMIT 3
        """
        df_merged = pd.read_sql(query, connection)
        
        # Check corresponding terms in original table
        for _, row in df_merged.iterrows():
            po_num = row['PO_NUM']
            cursor = connection.cursor()
            cursor.execute("""
            SELECT GROUP_CONCAT(TERM SEPARATOR '\n') as ORIGINAL_TERMS 
            FROM po_terms 
            WHERE PO_NUMBER = %s 
            GROUP BY PO_NUMBER
            """, (po_num,))
            result = cursor.fetchone()
            original_terms = result[0] if result else "No terms found"
            cursor.close()
            
            print(f"\n=== TERMS COMPARISON FOR PO {po_num} ===")
            print("\nORIGINAL TERMS from po_terms table:")
            print(original_terms)
            print("\nMERGED TERMS in po_followup_merged table:")
            print(row['TERMS'])
    except Error as e:
        print(f"Error checking terms merging: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    print("===== PO TABLES COMPARISON REPORT =====\n")
    
    # 1. Compare row counts between tables
    compare_table_row_counts()
    
    # 2. Compare sample data from both tables
    compare_column_data()
    
    # 3. Check for any missing PO numbers
    check_for_missing_data()
    
    # 4. Verify terms were properly merged
    check_terms_merging()
    
    print("\n===== END OF COMPARISON REPORT =====")
