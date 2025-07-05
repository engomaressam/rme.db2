import pymysql
import sys

# Database credentials
HOST = '10.10.11.242'
USER = 'omar2'
PASSWORD = 'Omar_54321'
DB = 'RME_TEST'
TABLE = 'po.pdfs'

# Connect to MySQL
def connect():
    try:
        conn = pymysql.connect(
            host=HOST,
            user=USER,
            password=PASSWORD,
            database=DB,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print('Connected to MySQL.')
        return conn
    except Exception as e:
        print(f'Failed to connect: {e}')
        sys.exit(1)

import csv

def preview_rows(conn, limit=120):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT id, extracted_text FROM `{TABLE}` ORDER BY id ASC LIMIT {limit}")
        rows = cursor.fetchall()
        # Print preview safely for terminal
        for row in rows:
            text = row['extracted_text']
            snippet = text[:80].replace('\n', ' ') if text else '[EMPTY]'
            try:
                print(f"ID: {row['id']}, Text: {snippet}".encode('utf-8', errors='replace').decode('cp1252', errors='replace'))
            except Exception:
                print(f"ID: {row['id']} (unprintable snippet)")
        # Save CSV for review
        with open('ocr_preview.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'extracted_text'])
            for row in rows:
                writer.writerow([row['id'], row['extracted_text']])
        print("\nSaved preview to ocr_preview.csv. Open this file in Excel or Notepad to review all text.")
    return rows

def delete_rows(conn, ids):
    with conn.cursor() as cursor:
        format_strings = ','.join(['%s'] * len(ids))
        cursor.execute(f"DELETE FROM `{TABLE}` WHERE id IN ({format_strings})", tuple(ids))
    conn.commit()
    print(f"Deleted {len(ids)} rows.")

def main():
    conn = connect()
    rows = preview_rows(conn)
    print("\nReview the above rows. Edit the script to specify which IDs to delete.")
    # Example: ids_to_delete = [1,2,3,...]
    ids_to_delete = []  # <-- Fill this list after review
    if ids_to_delete:
        delete_rows(conn, ids_to_delete)
    else:
        print("No rows deleted. Edit 'ids_to_delete' in the script after reviewing output.")
    conn.close()

if __name__ == '__main__':
    main()
