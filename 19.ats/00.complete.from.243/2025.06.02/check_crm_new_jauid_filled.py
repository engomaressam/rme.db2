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
            cur.execute("SELECT COUNT(*) FROM pdf_extracted_data WHERE crm_new_jauid IS NOT NULL")
            count = cur.fetchone()[0]
            print(f"Rows with crm_new_jauid filled: {count}")

if __name__ == "__main__":
    main() 