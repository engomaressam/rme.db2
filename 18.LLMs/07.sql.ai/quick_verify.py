import mysql.connector

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

try:
    # Connect to the database
    print("Connecting to database...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Check row counts
    print("\nCHECKING ROW COUNTS:")
    cursor.execute("SELECT COUNT(*) FROM RME_PO_Follow_Up_Report")
    original_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM po_followup_merged")
    merged_count = cursor.fetchone()[0]
    
    print(f"Original table: {original_count} rows")
    print(f"Merged table: {merged_count} rows")
    print(f"Row count match: {'✓' if original_count == merged_count else 'X'}")
    
    # Check random PO numbers to verify data consistency
    print("\nCHECKING DATA CONSISTENCY FOR 3 RANDOM POs:")
    cursor.execute("SELECT DISTINCT PO_NUM FROM po_followup_merged LIMIT 3")
    sample_pos = [row[0] for row in cursor.fetchall()]
    
    for po_num in sample_pos:
        print(f"\nChecking PO: {po_num}")
        
        # Check if this PO has the same number of rows in both tables
        cursor.execute(f"SELECT COUNT(*) FROM RME_PO_Follow_Up_Report WHERE PO_NUM = '{po_num}'")
        orig_po_count = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM po_followup_merged WHERE PO_NUM = '{po_num}'")
        merged_po_count = cursor.fetchone()[0]
        
        print(f"  Row count in original: {orig_po_count}")
        print(f"  Row count in merged: {merged_po_count}")
        print(f"  Counts match: {'✓' if orig_po_count == merged_po_count else 'X'}")
        
        # Sample a row to compare specific fields
        cursor.execute(f"""
            SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME 
            FROM RME_PO_Follow_Up_Report 
            WHERE PO_NUM = '{po_num}' LIMIT 1
        """)
        orig_sample = cursor.fetchone()
        
        cursor.execute(f"""
            SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME 
            FROM po_followup_merged 
            WHERE PO_NUM = '{po_num}' LIMIT 1
        """)
        merged_sample = cursor.fetchone()
        
        fields = ["PO_NUM", "PROJECT_NAME", "VENDOR_NAME"]
        match = True
        for i, field in enumerate(fields):
            field_match = orig_sample[i] == merged_sample[i]
            if not field_match:
                match = False
            status = "✓" if field_match else "X"
            print(f"  {field}: {status}")
        
        print(f"  Overall data match: {'✓' if match else 'X'}")
        
        # Check if this PO has terms
        cursor.execute(f"SELECT TERMS FROM po_followup_merged WHERE PO_NUM = '{po_num}' LIMIT 1")
        terms_result = cursor.fetchone()
        has_terms = terms_result and terms_result[0] is not None and terms_result[0].strip() != ""
        print(f"  Has terms in merged table: {'Yes' if has_terms else 'No'}")
    
    cursor.close()
    conn.close()
    
    print("\nVerification completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
