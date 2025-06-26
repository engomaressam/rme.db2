import psycopg2
import re
from datetime import datetime

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

def add_createdon_column():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'pdf_extracted_data' 
                        AND column_name = 'createdon'
                    ) THEN
                        ALTER TABLE pdf_extracted_data ADD COLUMN createdon DATE;
                    END IF;
                END $$;
            """)
            conn.commit()
    print("Added createdon column if missing.")

def extract_date_from_filename(filename):
    match = re.match(r"(\d{4}-\d{2}-\d{2})_", filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except Exception:
            return None
    return None

def populate_createdon():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, pdf_filename FROM pdf_extracted_data WHERE createdon IS NULL")
            rows = cur.fetchall()
            updated = 0
            for row_id, filename in rows:
                if not filename:
                    continue
                date_val = extract_date_from_filename(filename)
                if date_val:
                    cur.execute("UPDATE pdf_extracted_data SET createdon = %s WHERE id = %s", (date_val, row_id))
                    updated += 1
            conn.commit()
    print(f"Populated createdon for {updated} rows.")

def main():
    add_createdon_column()
    populate_createdon()

if __name__ == "__main__":
    main() 