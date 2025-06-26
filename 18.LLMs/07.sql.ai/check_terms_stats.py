import mysql.connector

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

try:
    # Connect to database
    print("Connecting to database...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Get total PO count in merged table
    print("Calculating statistics on terms data...")
    cursor.execute("SELECT COUNT(*) FROM po_followup_merged")
    total_count = cursor.fetchone()[0]
    
    # Count distinct POs with terms
    cursor.execute("""
        SELECT COUNT(DISTINCT PO_NUM) 
        FROM po_followup_merged 
        WHERE TERMS IS NOT NULL AND TRIM(TERMS) != ''
    """)
    pos_with_terms = cursor.fetchone()[0]
    
    # Count distinct POs without terms
    cursor.execute("""
        SELECT COUNT(DISTINCT PO_NUM) 
        FROM po_followup_merged 
        WHERE TERMS IS NULL OR TRIM(TERMS) = ''
    """)
    pos_without_terms = cursor.fetchone()[0]
    
    # Total unique PO numbers
    cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged")
    total_unique_pos = cursor.fetchone()[0]
    
    # Calculate percentages
    percent_with_terms = (pos_with_terms / total_unique_pos) * 100
    percent_without_terms = (pos_without_terms / total_unique_pos) * 100
    
    # Check if source terms table has any terms at all
    cursor.execute("SELECT COUNT(*) FROM po_terms")
    total_terms_count = cursor.fetchone()[0]
    
    # Count distinct PO numbers in source terms table
    cursor.execute("SELECT COUNT(DISTINCT PO_NUMBER) FROM po_terms")
    unique_pos_with_terms = cursor.fetchone()[0]
    
    print("\n==== TERMS INTEGRATION STATISTICS ====")
    print(f"Total rows in merged table: {total_count:,}")
    print(f"Total unique POs in merged table: {total_unique_pos:,}")
    print(f"POs with terms: {pos_with_terms:,} ({percent_with_terms:.2f}%)")
    print(f"POs without terms: {pos_without_terms:,} ({percent_without_terms:.2f}%)")
    
    print("\n==== SOURCE TERMS TABLE STATISTICS ====")
    print(f"Total rows in po_terms table: {total_terms_count:,}")
    print(f"Unique PO numbers with terms in source table: {unique_pos_with_terms:,}")
    
    # Check percentage of POs that should have terms but don't
    matching_rate = (pos_with_terms / unique_pos_with_terms) * 100 if unique_pos_with_terms > 0 else 0
    print(f"\nPercentage of POs with terms in source that have terms in merged table: {matching_rate:.2f}%")
    
    if matching_rate < 100 and unique_pos_with_terms > 0:
        print("\nCHECKING FOR MISSING TERMS:")
        # Find some examples of POs that should have terms but don't
        cursor.execute("""
            SELECT pt.PO_NUMBER 
            FROM po_terms pt
            LEFT JOIN (
                SELECT DISTINCT PO_NUM 
                FROM po_followup_merged 
                WHERE TERMS IS NOT NULL AND TRIM(TERMS) != ''
            ) pm ON pt.PO_NUMBER = pm.PO_NUM
            WHERE pm.PO_NUM IS NULL
            GROUP BY pt.PO_NUMBER
            LIMIT 5
        """)
        missing_pos = cursor.fetchall()
        if missing_pos:
            print("Example POs with terms in source but missing in merged table:")
            for po in missing_pos:
                print(f"  - {po[0]}")
    
    cursor.close()
    conn.close()
    
    print("\nAnalysis completed!")
    
except Exception as e:
    print(f"Error: {e}")
