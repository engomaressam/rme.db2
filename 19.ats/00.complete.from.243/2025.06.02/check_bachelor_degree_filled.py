import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

def main():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pdf_extracted_data WHERE bachelor_degree IS NOT NULL AND bachelor_degree <> ''")
            count = cur.fetchone()[0]
            print(f"Rows with bachelor_degree filled: {count}")

if __name__ == "__main__":
    main() 