import mysql.connector
import sys

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
    except Exception as e:
        print(f"Database error: {e}")
        return None

# Clear screen with separator
print("=" * 80)
print("DATA VERIFICATION REPORT")
print("=" * 80)

# 1. Row count verification
row_count_query = """
SELECT 
    (SELECT COUNT(*) FROM RME_PO_Follow_Up_Report) AS original_count,
    (SELECT COUNT(*) FROM po_followup_merged) AS merged_count
"""

counts = execute_query(row_count_query)[0]
original_count = counts[0]
merged_count = counts[1]

print(f"Original table rows: {original_count}")
print(f"Merged table rows:   {merged_count}")
print(f"Match: {'✓' if original_count == merged_count else '✗'}")
print("-" * 80)

# 2. Sample PO with terms
sample_po_query = """
SELECT PO_NUM FROM po_followup_merged 
WHERE TERMS IS NOT NULL AND TERMS != '' 
LIMIT 1
"""

sample_po_result = execute_query(sample_po_query)
if sample_po_result and len(sample_po_result) > 0:
    sample_po = sample_po_result[0][0]
    print(f"Sample PO with terms: {sample_po}")
    
    # 3. Compare data for this PO
    print("\nCOMPARISON OF DATA FOR SAMPLE PO")
    print("-" * 80)
    
    # Select a sample row to compare between tables
    original_data_query = f"""
    SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME, ITEM_DESCRIPTION, UNIT_PRICE 
    FROM RME_PO_Follow_Up_Report 
    WHERE PO_NUM = %s
    LIMIT 1
    """
    
    merged_data_query = f"""
    SELECT PO_NUM, PROJECT_NAME, VENDOR_NAME, ITEM_DESCRIPTION, UNIT_PRICE, TERMS 
    FROM po_followup_merged 
    WHERE PO_NUM = %s
    LIMIT 1
    """
    
    original_row = execute_query(original_data_query, (sample_po,))[0]
    merged_row = execute_query(merged_data_query, (sample_po,))[0]
    
    fields = ["PO_NUM", "PROJECT_NAME", "VENDOR_NAME", "ITEM_DESCRIPTION", "UNIT_PRICE"]
    print("Field comparison (original vs merged):")
    match = True
    for i, field in enumerate(fields):
        orig_value = str(original_row[i]) if original_row[i] is not None else "NULL"
        merged_value = str(merged_row[i]) if merged_row[i] is not None else "NULL"
        field_match = orig_value == merged_value
        if not field_match:
            match = False
        print(f"{field}: {'✓' if field_match else '✗'} - Values {'' if field_match else 'do not '}match")
    
    print(f"\nOverall data match: {'✓' if match else '✗'}")
    print("-" * 80)
    
    # 4. Check terms data
    original_terms_query = f"""
    SELECT PO_NUMBER, TERM FROM po_terms 
    WHERE PO_NUMBER = %s
    """
    
    original_terms = execute_query(original_terms_query, (sample_po,))
    merged_terms = merged_row[5]  # TERMS is the 6th field in the merged query
    
    print("\nTERMS VERIFICATION")
    print("-" * 80)
    
    print("Original terms from po_terms table:")
    if original_terms and len(original_terms) > 0:
        for i, term_row in enumerate(original_terms):
            print(f"  {i+1}. {term_row[1]}")
        
        print("\nMerged terms in po_followup_merged table:")
        if merged_terms:
            for i, term_line in enumerate(merged_terms.split('\n')):
                if term_line.strip():
                    print(f"  {i+1}. {term_line}")
            
            # Check if all original terms are included in merged terms
            all_found = True
            missing_terms = []
            for term_row in original_terms:
                term = term_row[1]
                if term not in merged_terms:
                    all_found = False
                    missing_terms.append(term)
            
            print(f"\nAll original terms found in merged terms: {'✓' if all_found else '✗'}")
            if not all_found:
                print("Missing terms:")
                for term in missing_terms:
                    print(f"  - {term}")
        else:
            print("  No merged terms found")
    else:
        print("  No original terms found")
    
    print("-" * 80)
    
    # 5. Row count comparison for this PO
    count_query = f"""
    SELECT 
        (SELECT COUNT(*) FROM RME_PO_Follow_Up_Report WHERE PO_NUM = %s) AS original_po_count,
        (SELECT COUNT(*) FROM po_followup_merged WHERE PO_NUM = %s) AS merged_po_count
    """
    
    po_counts = execute_query(count_query, (sample_po, sample_po))[0]
    original_po_count = po_counts[0]
    merged_po_count = po_counts[1]
    
    print(f"Row counts for PO {sample_po}:")
    print(f"  Original table: {original_po_count}")
    print(f"  Merged table:   {merged_po_count}")
    print(f"  Match: {'✓' if original_po_count == merged_po_count else '✗'}")
else:
    print("No PO with terms found in merged table")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
