import mysql.connector
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
    except Exception as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def compare_sample_data():
    """Compare sample data between tables"""
    connection = connect_to_database()
    if not connection:
        return
    
    try:
        # Get a sample PO_NUM to use for comparison
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT PO_NUM FROM po_followup_merged WHERE TERMS IS NOT NULL LIMIT 1")
        sample_po = cursor.fetchone()[0]
        cursor.close()
        
        print(f"Comparing sample data for PO_NUM: {sample_po}")
        
        # Compare data for this PO in both tables
        query_orig = f"""
        SELECT id, PO_NUM, COMMENTS, APPROVED_DATE, UOM, 
               ITEM_DESCRIPTION, UNIT_PRICE, QUANTITY_RECEIVED, 
               LINE_AMOUNT, PROJECT_NAME, VENDOR_NAME 
        FROM RME_PO_Follow_Up_Report 
        WHERE PO_NUM = '{sample_po}'
        LIMIT 5
        """
        df_orig = pd.read_sql(query_orig, connection)
        
        query_merged = f"""
        SELECT id, PO_NUM, COMMENTS, APPROVED_DATE, UOM, 
               ITEM_DESCRIPTION, UNIT_PRICE, QUANTITY_RECEIVED, 
               LINE_AMOUNT, PROJECT_NAME, VENDOR_NAME, TERMS 
        FROM po_followup_merged 
        WHERE PO_NUM = '{sample_po}'
        LIMIT 5
        """
        df_merged = pd.read_sql(query_merged, connection)
        
        print("\n=== ORIGINAL TABLE DATA ===")
        print(tabulate(df_orig, headers='keys', tablefmt='psql', showindex=False))
        
        print("\n=== MERGED TABLE DATA (includes TERMS) ===")
        print(tabulate(df_merged, headers='keys', tablefmt='psql', showindex=False))
        
        # Check original terms from po_terms table
        query_terms = f"""
        SELECT PO_NUMBER, TERM FROM po_terms WHERE PO_NUMBER = '{sample_po}'
        """
        df_terms = pd.read_sql(query_terms, connection)
        
        print("\n=== ORIGINAL TERMS FROM po_terms TABLE ===")
        if not df_terms.empty:
            print(tabulate(df_terms, headers='keys', tablefmt='psql', showindex=False))
            
            # Check if terms were properly merged
            all_terms = '\n'.join(df_terms['TERM'].tolist())
            merged_terms = df_merged['TERMS'].iloc[0] if not df_merged.empty else ""
            
            print("\n=== TERMS VERIFICATION ===")
            if all_terms.strip() == merged_terms.strip():
                print("✓ TERMS were correctly merged (exact match)")
            else:
                print("⚠ TERMS may not match exactly, showing both for comparison:")
                print("\nOriginal concatenated terms:")
                print(all_terms)
                print("\nMerged terms:")
                print(merged_terms)
        else:
            print("No terms found for this PO in the po_terms table.")
            
    except Exception as e:
        print(f"Error comparing sample data: {e}")
    finally:
        if connection:
            connection.close()

if __name__ == "__main__":
    compare_sample_data()
