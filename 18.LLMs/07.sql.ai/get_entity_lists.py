import psycopg2
import os
from dotenv import load_dotenv

def get_lists():
    """Connects to the database and fetches unique project and supplier lists."""
    load_dotenv()

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PG_DB"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT")
        )
    except psycopg2.OperationalError as e:
        print(f"Database connection failed: {e}")
        print("Please ensure your .env file is configured correctly and PostgreSQL is running.")
        return

    print("Successfully connected to the database.")

    with conn.cursor() as cur:
        # Fetch unique, non-null, non-empty project names
        cur.execute('SELECT DISTINCT \"PROJECT_NAME\" FROM po_followup_rev19 WHERE \"PROJECT_NAME\" IS NOT NULL AND \"PROJECT_NAME\" <> \'\';')
        projects = sorted([row[0] for row in cur.fetchall()])

        # Fetch unique, non-null, non-empty supplier names
        cur.execute('SELECT DISTINCT \"VENDOR_NAME\" FROM po_followup_rev19 WHERE \"VENDOR_NAME\" IS NOT NULL AND \"VENDOR_NAME\" <> \'\';')
        suppliers = sorted([row[0] for row in cur.fetchall()])

    conn.close()

    print("\n--- UNIQUE PROJECTS ---")
    print("UNIQUE_PROJECTS = [")
    for project in projects:
                print(f"    {repr(project)},")
    print("]")

    print("\n--- UNIQUE SUPPLIERS ---")
    print("UNIQUE_SUPPLIERS = [")
    for supplier in suppliers:
                print(f"    {repr(supplier)},")
    print("]")

if __name__ == "__main__":
    get_lists()
