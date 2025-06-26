import mysql.connector

DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def main():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        print("Dropping existing table if exists...")
        cursor.execute("DROP TABLE IF EXISTS po_followup_merged;")
        print("Creating new table po_followup_merged...")
        cursor.execute("""
            CREATE TABLE po_followup_merged (
                id INT AUTO_INCREMENT PRIMARY KEY,
                PO_NUM VARCHAR(255),
                COMMENTS TEXT,
                APPROVED_DATE DATE,
                UOM VARCHAR(50),
                ITEM_DESCRIPTION TEXT,
                UNIT_PRICE DECIMAL(18,2),
                QUANTITY_RECEIVED DECIMAL(18,2),
                LINE_AMOUNT DECIMAL(18,2),
                PROJECT_NAME VARCHAR(255),
                VENDOR_NAME VARCHAR(255),
                TERMS TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        print("Fetching merged terms from po_terms...")
        cursor.execute("""
            SELECT PO_NUMBER, GROUP_CONCAT(LONG_TEXT SEPARATOR '\n') AS MERGED_TERMS
            FROM po_terms
            GROUP BY PO_NUMBER
        """)
        terms_dict = {row[0]: row[1] for row in cursor.fetchall()}
        print(f"Fetched {len(terms_dict)} unique PO_NUMBERs with merged terms.")
        print("Fetching data from RME_PO_Follow_Up_Report...")
        cursor.execute("""
            SELECT
                PO_NUM,
                COMMENTS,
                APPROVED_DATE,
                UOM,
                ITEM_DESCRIPTION,
                UNIT_PRICE,
                QUANTITY_RECEIVED,
                LINE_AMOUNT,
                PROJECT_NAME,
                VENDOR_NAME
            FROM RME_PO_Follow_Up_Report
        """)
        rows = cursor.fetchall()
        print(f"Fetched {len(rows)} rows from RME_PO_Follow_Up_Report.")
        insert_sql = """
            INSERT INTO po_followup_merged (
                PO_NUM, COMMENTS, APPROVED_DATE, UOM, ITEM_DESCRIPTION, UNIT_PRICE,
                QUANTITY_RECEIVED, LINE_AMOUNT, PROJECT_NAME, VENDOR_NAME, TERMS
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        count = 0
        for row in rows:
            po_num = row[0]
            merged_terms = terms_dict.get(po_num, "")
            cursor.execute(insert_sql, row + (merged_terms,))
            count += 1
            if count % 1000 == 0:
                print(f"Inserted {count} rows...")
        conn.commit()
        print(f"Table po_followup_merged created and populated successfully with {count} rows.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
