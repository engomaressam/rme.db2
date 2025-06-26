import mysql.connector

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

# Connect to database
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Get total row count
cursor.execute("SELECT COUNT(*) FROM po_followup_merged")
total_rows = cursor.fetchone()[0]
print(f"Total rows in merged table: {total_rows}")

# Count rows with NULL terms
cursor.execute("SELECT COUNT(*) FROM po_followup_merged WHERE TERMS IS NULL OR TRIM(TERMS) = ''")
null_terms_count = cursor.fetchone()[0]
print(f"Rows with NULL terms: {null_terms_count}")

# Count rows with non-NULL terms
cursor.execute("SELECT COUNT(*) FROM po_followup_merged WHERE TERMS IS NOT NULL AND TRIM(TERMS) != ''")
non_null_terms_count = cursor.fetchone()[0]
print(f"Rows with terms: {non_null_terms_count}")

# Calculate percentages
null_percent = (null_terms_count / total_rows) * 100
non_null_percent = (non_null_terms_count / total_rows) * 100

print(f"Percentage with NULL terms: {null_percent:.2f}%")
print(f"Percentage with terms: {non_null_percent:.2f}%")

# Count unique PO numbers
cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged")
unique_pos = cursor.fetchone()[0]
print(f"\nUnique PO numbers: {unique_pos}")

# Count unique PO numbers with terms
cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged WHERE TERMS IS NOT NULL AND TRIM(TERMS) != ''")
unique_pos_with_terms = cursor.fetchone()[0]
print(f"Unique PO numbers with terms: {unique_pos_with_terms}")

# Count unique PO numbers without terms
cursor.execute("SELECT COUNT(DISTINCT PO_NUM) FROM po_followup_merged WHERE TERMS IS NULL OR TRIM(TERMS) = ''")
unique_pos_without_terms = cursor.fetchone()[0]
print(f"Unique PO numbers without terms: {unique_pos_without_terms}")

# Calculate percentages for unique POs
unique_with_terms_percent = (unique_pos_with_terms / unique_pos) * 100
unique_without_terms_percent = (unique_pos_without_terms / unique_pos) * 100

print(f"Percentage of unique POs with terms: {unique_with_terms_percent:.2f}%")
print(f"Percentage of unique POs without terms: {unique_without_terms_percent:.2f}%")

# Close connection
cursor.close()
conn.close()
