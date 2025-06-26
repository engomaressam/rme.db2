import mysql.connector

# Connect to database
print("Connecting to database...")
conn = mysql.connector.connect(
    host='10.10.11.242',
    user='omar2',
    password='Omar_54321',
    database='RME_TEST'
)

# Create cursor
cursor = conn.cursor()

# Get total row count
print("Calculating statistics...")
cursor.execute("SELECT COUNT(*) FROM po_followup_merged")
total_rows = cursor.fetchone()[0]

# Count rows with NULL or empty terms
cursor.execute("SELECT COUNT(*) FROM po_followup_merged WHERE TERMS IS NULL OR TRIM(TERMS) = ''")
null_terms_count = cursor.fetchone()[0]

# Calculate non-null terms count
non_null_terms_count = total_rows - null_terms_count

# Calculate percentages
null_percent = (null_terms_count / total_rows) * 100
non_null_percent = 100 - null_percent

# Count unique PO numbers
cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged")
unique_pos = cursor.fetchone()[0]

# Count unique PO numbers with terms
cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged WHERE TERMS IS NOT NULL AND TRIM(TERMS) != ''")
unique_pos_with_terms = cursor.fetchone()[0]

# Count unique PO numbers without terms
unique_pos_without_terms = unique_pos - unique_pos_with_terms

# Calculate percentages for unique POs
unique_with_terms_percent = (unique_pos_with_terms / unique_pos) * 100
unique_without_terms_percent = 100 - unique_with_terms_percent

# Print results
print("\n=== TERMS STATISTICS ===")
print(f"Total rows in merged table: {total_rows:,}")
print(f"Rows with terms: {non_null_terms_count:,} ({non_null_percent:.2f}%)")
print(f"Rows without terms: {null_terms_count:,} ({null_percent:.2f}%)")

print("\n=== UNIQUE PO STATISTICS ===")
print(f"Total unique PO numbers: {unique_pos:,}")
print(f"Unique POs with terms: {unique_pos_with_terms:,} ({unique_with_terms_percent:.2f}%)")
print(f"Unique POs without terms: {unique_pos_without_terms:,} ({unique_without_terms_percent:.2f}%)")

# Additional check: How many terms were in original terms table?
cursor.execute("SELECT COUNT(DISTINCT PO_NUMBER) FROM po_terms")
source_unique_pos = cursor.fetchone()[0]
print(f"\nUnique POs in source terms table: {source_unique_pos:,}")

coverage_rate = (unique_pos_with_terms / source_unique_pos) * 100 if source_unique_pos > 0 else 0
print(f"Terms coverage rate: {coverage_rate:.2f}%")

# Close connections
cursor.close()
conn.close()
