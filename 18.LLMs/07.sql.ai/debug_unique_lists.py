import psycopg2
import sys

# Compare the unique project lists between rev.16 and rev.20 databases

# Rev.16 database config
REV16_DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'po_data',
    'table': 'po_followup_merged'
}

# Rev.20 database config
REV20_DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public',
    'table': 'po_followup_rev19'
}

def fetch_unique_list(config, column):
    connection = None
    try:
        connection = psycopg2.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT DISTINCT "{column}" 
            FROM {config['schema']}.{config['table']} 
            WHERE "{column}" IS NOT NULL AND "{column}" != '' 
            ORDER BY "{column}"
        """)
        results = [row[0] for row in cursor.fetchall() if row[0]]
        return results
    except Exception as e:
        print(f"Error fetching unique {column} from {config['schema']}.{config['table']}: {e}")
        return []
    finally:
        if connection:
            connection.close()

def print_list_stats(name, items):
    print(f"{name} count: {len(items)}")
    print(f"First 5 items: {items[:5]}")
    print(f"Last 5 items: {items[-5:]}")
    print("-" * 50)

if __name__ == "__main__":
    # Fetch project lists
    rev16_projects = fetch_unique_list(REV16_DB_CONFIG, "PROJECT_NAME")
    rev20_projects = fetch_unique_list(REV20_DB_CONFIG, "PROJECT_NAME")
    
    print("UNIQUE PROJECT LISTS COMPARISON")
    print("=" * 50)
    print_list_stats("Rev.16 projects", rev16_projects)
    print_list_stats("Rev.20 projects", rev20_projects)
    
    # Find projects in rev.16 but not in rev.20
    missing_in_rev20 = [p for p in rev16_projects if p not in rev20_projects]
    print(f"Projects in rev.16 but missing in rev.20: {len(missing_in_rev20)}")
    if missing_in_rev20:
        print("Examples:", missing_in_rev20[:10])
    
    # Find projects in rev.20 but not in rev.16
    new_in_rev20 = [p for p in rev20_projects if p not in rev16_projects]
    print(f"Projects in rev.20 but not in rev.16: {len(new_in_rev20)}")
    if new_in_rev20:
        print("Examples:", new_in_rev20[:10])
