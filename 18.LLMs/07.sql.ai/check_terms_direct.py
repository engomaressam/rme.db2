import mysql.connector

try:
    # Connect to database
    print("Connecting to database...")
    conn = mysql.connector.connect(
        host='10.10.11.242',
        user='omar2',
        password='Omar_54321',
        database='RME_TEST'
    )
    
    cursor = conn.cursor(buffered=True)
    
    # Execute direct queries to get only what we need
    print("Running analysis...")
    
    # 1. Count total distinct PO numbers
    cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged")
    total_pos = cursor.fetchone()[0]
    print(f"Total unique PO numbers: {total_pos}")
    
    # 2. Count POs with terms
    cursor.execute("""
        SELECT COUNT(DISTINCT PO_NUM) 
        FROM po_followup_merged 
        WHERE TERMS IS NOT NULL AND TERMS != ''
    """)
    pos_with_terms = cursor.fetchone()[0]
    print(f"POs with terms: {pos_with_terms}")
    
    # 3. Calculate percentages
    percent_with_terms = (pos_with_terms / total_pos) * 100
    percent_without_terms = 100 - percent_with_terms
    
    print(f"Percentage of POs with terms: {percent_with_terms:.2f}%")
    print(f"Percentage of POs without terms: {percent_without_terms:.2f}%")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
