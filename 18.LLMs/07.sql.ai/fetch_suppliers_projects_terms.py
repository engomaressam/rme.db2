import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def fetch_examples():
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    # Get some suppliers with their terms
    cursor.execute("SELECT DISTINCT vendor_name, terms FROM po_followup_with_terms WHERE vendor_name IS NOT NULL AND terms IS NOT NULL AND terms != '' LIMIT 5")
    suppliers = cursor.fetchall()
    # Get some projects
    cursor.execute("SELECT DISTINCT project_name FROM po_followup_with_terms WHERE project_name IS NOT NULL AND project_name != '' LIMIT 5")
    projects = [row[0] for row in cursor.fetchall()]
    cursor.close()
    connection.close()
    return suppliers, projects

def write_example_questions_utf8(filename="questions_utf8.txt"):
    suppliers, projects = fetch_examples()
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Sample supplier/terms questions:\n")
        for supplier, terms in suppliers:
            f.write(f"What are the terms for supplier {supplier}?\n")
            f.write(f"Show all POs for supplier {supplier} and their terms.\n")
            f.write(f"List PO numbers and terms for {supplier}.\n")
        f.write("\nSample project questions:\n")
        for project in projects:
            f.write(f"What are the terms for project {project}?\n")
            f.write(f"Show all POs for project {project} and their terms.\n")
            f.write(f"List PO numbers and terms for project {project}.\n")

if __name__ == "__main__":
    write_example_questions_utf8()
