import mysql.connector

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

def main():
    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    cursor = conn.cursor()

    print("Dropping po_followup_with_terms if exists...")
    cursor.execute("DROP TABLE IF EXISTS po_followup_with_terms")
    print("Creating po_followup_with_terms with SQL join...")
    cursor.execute("""
        CREATE TABLE po_followup_with_terms AS
        SELECT
            f.*,
            t.LONG_TEXT
        FROM
            po_followup_for_chatbot f
        LEFT JOIN
            po_terms t
        ON
            f.po_number = t.PO_NUMBER
    """)
    conn.commit()
    print("Table created.")

    print("Getting stats...")
    cursor.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN LONG_TEXT IS NOT NULL THEN 1 ELSE 0 END) AS with_terms,
            SUM(CASE WHEN LONG_TEXT IS NULL THEN 1 ELSE 0 END) AS without_terms
        FROM
            po_followup_with_terms
    """)
    result = cursor.fetchone()
    print(f"Total: {result[0]}, POs with terms: {result[1]}, POs without terms: {result[2]}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main() 