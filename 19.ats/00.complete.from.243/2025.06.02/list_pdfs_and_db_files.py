import os
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

PDF_DIR = r'C:\cvs'

def list_dir_pdfs():
    files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    with open('dir_pdfs.txt', 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file + '\n')
    print(f"Wrote {len(files)} PDF filenames from directory to dir_pdfs.txt")

def list_db_pdfs():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT pdf_filename FROM pdf_extracted_data')
            files = [row[0] for row in cur.fetchall() if row[0]]
    with open('db_pdfs.txt', 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file + '\n')
    print(f"Wrote {len(files)} PDF filenames from database to db_pdfs.txt")

def main():
    list_dir_pdfs()
    list_db_pdfs()

if __name__ == "__main__":
    main() 