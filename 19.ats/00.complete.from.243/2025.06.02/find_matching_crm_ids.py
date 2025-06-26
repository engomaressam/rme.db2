import psycopg2
import requests
from requests_ntlm import HttpNtlmAuth

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

# CRM API Configuration
CRM_URL = "https://rmecrm.rowad-rme.com/RMECRM/api/data/v8.2"
USERNAME = "Rowad\\Omar Essam"
PASSWORD = "PMO@1234"

SAMPLE_SIZE = 1000  # Increase for better overlap

def get_crm_ids_from_db():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT crm_applicationid FROM pdf_extracted_data WHERE crm_applicationid IS NOT NULL")
            return set(row[0] for row in cur.fetchall())

def get_crm_ids_from_crm():
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    })
    url = f"{CRM_URL}/new_jobapplications?$select=new_jobapplicationid&$top={SAMPLE_SIZE}"
    response = session.get(url)
    response.raise_for_status()
    data = response.json().get('value', [])
    return set(item.get('new_jobapplicationid') for item in data if item.get('new_jobapplicationid'))

def main():
    db_ids = get_crm_ids_from_db()
    crm_ids = get_crm_ids_from_crm()
    matches = db_ids & crm_ids
    print(f"Total DB IDs: {len(db_ids)}")
    print(f"Total CRM IDs: {len(crm_ids)}")
    print(f"Matching IDs: {len(matches)}")
    if matches:
        print("Sample matching IDs:")
        for i, id in enumerate(list(matches)[:10], 1):
            print(f"{i}: {id}")
    else:
        print("No matching IDs found.")

if __name__ == "__main__":
    main() 