import mysql.connector

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def connect_and_execute(query):
    """Connect to the database and execute a query"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

print("========== PO TABLE VERIFICATION ==========")

# 1. Compare row counts
original_count = connect_and_execute("SELECT COUNT(*) FROM RME_PO_Follow_Up_Report")[0][0]
merged_count = connect_and_execute("SELECT COUNT(*) FROM po_followup_merged")[0][0]

print(f"Original table (RME_PO_Follow_Up_Report): {original_count} rows")
print(f"Merged table (po_followup_merged): {merged_count} rows")
print(f"Difference: {original_count - merged_count} rows")

# 2. Check if all PO numbers are preserved
missing_po_count = connect_and_execute("""
    SELECT COUNT(DISTINCT po1.PO_NUM) 
    FROM RME_PO_Follow_Up_Report po1 
    WHERE NOT EXISTS (
        SELECT 1 FROM po_followup_merged po2 
        WHERE po1.PO_NUM = po2.PO_NUM
    )
""")[0][0]

print(f"\nMissing PO numbers in merged table: {missing_po_count}")

# 3. Get a sample merged row with terms to verify structure
print("\n=== SAMPLE ROW FROM MERGED TABLE (With Terms) ===")
sample_row = connect_and_execute("""
    SELECT * FROM po_followup_merged 
    WHERE TERMS IS NOT NULL AND LENGTH(TERMS) > 10
    LIMIT 1
""")

if sample_row and len(sample_row) > 0:
    # Get column names
    col_query = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'RME_TEST' AND TABLE_NAME = 'po_followup_merged'"
    columns = [col[0] for col in connect_and_execute(col_query)]
    
    # Print sample row with column names
    sample_data = sample_row[0]
    po_num = None
    for i, col in enumerate(columns):
        value = sample_data[i] if i < len(sample_data) else None
        print(f"{col}: {value}")
        if col == "PO_NUM":
            po_num = value
    
    # 4. Compare with original terms if we found a PO number
    if po_num:
        print(f"\n=== ORIGINAL TERMS FOR PO_NUM {po_num} ===")
        original_terms = connect_and_execute(f"""
            SELECT PO_NUMBER, TERM FROM po_terms WHERE PO_NUMBER = '{po_num}'
        """)
        
        if original_terms and len(original_terms) > 0:
            for i, term_row in enumerate(original_terms):
                print(f"Term {i+1}: {term_row[1]}")
        else:
            print("No original terms found for this PO number.")
else:
    print("No rows with terms found in the merged table.")
