import mysql.connector
from mysql.connector import Error

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def execute_query(query, params=None):
    """Execute a query and return results"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
    except Error as e:
        print(f"Database error: {e}")
        return None

print("===== PO TABLE VERIFICATION =====")

# 1. Compare row counts
print("\n1. ROW COUNT COMPARISON:")
original_count = execute_query("SELECT COUNT(*) FROM RME_PO_Follow_Up_Report")[0][0]
merged_count = execute_query("SELECT COUNT(*) FROM po_followup_merged")[0][0]
print(f"Original table: {original_count} rows")
print(f"Merged table: {merged_count} rows")
print(f"Row count match: {'Yes' if original_count == merged_count else 'No'}")

# 2. Get a PO number with terms
print("\n2. FINDING A PO WITH TERMS:")
po_with_terms = execute_query("SELECT PO_NUM FROM po_followup_merged WHERE TERMS IS NOT NULL LIMIT 1")
if po_with_terms and len(po_with_terms) > 0:
    po_num = po_with_terms[0][0]
    print(f"Found PO number with terms: {po_num}")
    
    # 3. Check data consistency for this PO
    print(f"\n3. DATA COMPARISON FOR PO {po_num}:")
    
    # Get data from original table
    original_data = execute_query(f"""
        SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME, ITEM_DESCRIPTION 
        FROM RME_PO_Follow_Up_Report 
        WHERE PO_NUM = %s
        LIMIT 3
    """, (po_num,))
    
    # Get data from merged table
    merged_data = execute_query(f"""
        SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME, ITEM_DESCRIPTION, TERMS 
        FROM po_followup_merged 
        WHERE PO_NUM = %s
        LIMIT 3
    """, (po_num,))
    
    print("\nOriginal table data:")
    for i, row in enumerate(original_data):
        print(f"  Row {i+1}: PO_NUM={row[0]}, PROJECT={row[1][:20]}..., VENDOR={row[2][:20]}...")
    
    print("\nMerged table data:")
    for i, row in enumerate(merged_data):
        print(f"  Row {i+1}: PO_NUM={row[0]}, PROJECT={row[1][:20]}..., VENDOR={row[2][:20]}..., Has Terms: {'Yes' if row[4] else 'No'}")
    
    # 4. Check original terms
    print(f"\n4. ORIGINAL TERMS FROM po_terms TABLE FOR PO {po_num}:")
    terms_data = execute_query(f"""
        SELECT TERM FROM po_terms WHERE PO_NUMBER = %s
    """, (po_num,))
    
    if terms_data and len(terms_data) > 0:
        for i, term in enumerate(terms_data):
            print(f"  Term {i+1}: {term[0][:50]}...")
    else:
        print("  No terms found in po_terms table")
    
    # 5. Check merged terms
    print(f"\n5. MERGED TERMS IN po_followup_merged TABLE FOR PO {po_num}:")
    merged_terms = execute_query(f"""
        SELECT TERMS FROM po_followup_merged WHERE PO_NUM = %s LIMIT 1
    """, (po_num,))
    
    if merged_terms and len(merged_terms) > 0 and merged_terms[0][0]:
        terms_text = merged_terms[0][0]
        terms_lines = terms_text.split('\n')
        for i, line in enumerate(terms_lines[:3]):  # Show first 3 lines only
            print(f"  Line {i+1}: {line[:50]}...")
        if len(terms_lines) > 3:
            print(f"  ... {len(terms_lines)-3} more lines ...")
    else:
        print("  No merged terms found")
    
    # 6. Check data sample counts for this PO
    print(f"\n6. ROW COUNT FOR PO {po_num} IN BOTH TABLES:")
    original_po_count = execute_query(f"""
        SELECT COUNT(*) FROM RME_PO_Follow_Up_Report WHERE PO_NUM = %s
    """, (po_num,))[0][0]
    
    merged_po_count = execute_query(f"""
        SELECT COUNT(*) FROM po_followup_merged WHERE PO_NUM = %s
    """, (po_num,))[0][0]
    
    print(f"  Original table: {original_po_count} rows")
    print(f"  Merged table: {merged_po_count} rows")
    print(f"  Same number of rows: {'Yes' if original_po_count == merged_po_count else 'No'}")
else:
    print("No PO numbers with terms found in the merged table")

print("\n===== VERIFICATION COMPLETE =====")
