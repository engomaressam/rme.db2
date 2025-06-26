import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'options': '-c search_path=public,po_data'
}

def fetch_unique_projects():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT project_name FROM po_followup_rev19 ORDER BY project_name')
    projects = [row[0] for row in cur.fetchall()]
    print(f"Found {len(projects)} unique projects:")
    for p in projects[:20]:
        print(p)
    conn.close()

if __name__ == '__main__':
    fetch_unique_projects()
