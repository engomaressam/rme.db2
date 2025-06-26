import os
import requests
from requests_ntlm import HttpNtlmAuth
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CRM_URL = "https://rmecrm.rowad-rme.com/RMECRM/api/data/v8.2"
USERNAME = "Rowad\\Omar Essam"
PASSWORD = "PMO@1234"

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

# Fetch first 10 normalized filenames from CRM
def get_crm_normalized_filenames(limit=10):
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    })
    url = f"{CRM_URL}/new_jobapplications?$select=new_jobapplicationid,createdon&$top=50"
    response = session.get(url)
    response.raise_for_status()
    applications = response.json().get('value', [])
    crm_filenames = []
    for app in applications:
        app_id = app.get('new_jobapplicationid')
        createdon = app.get('createdon')
        if not (app_id and createdon):
            continue
        created_date = str(createdon).split('T')[0]
        # Fetch annotations for this application
        annotations_url = f"{CRM_URL}/annotations?$filter=_objectid_value eq {app_id}&$select=filename"
        ann_response = session.get(annotations_url)
        ann_response.raise_for_status()
        annotations = ann_response.json().get('value', [])
        for annotation in annotations:
            filename = annotation.get('filename')
            if filename:
                constructed_filename = f"{created_date}_{filename}"
                normalized = constructed_filename.replace(' ', '').lower()
                crm_filenames.append(normalized)
                if len(crm_filenames) >= limit:
                    return crm_filenames
    return crm_filenames

# Fetch first 10 normalized filenames from DB
def get_db_normalized_filenames(limit=10):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pdf_filename FROM pdf_extracted_data LIMIT %s", (limit,))
            rows = cur.fetchall()
            db_filenames = [row[0] for row in rows if row[0]]
            normalized = [fn.replace(' ', '').lower() for fn in db_filenames]
            return normalized

def main():
    print("[COMPARE] First 10 normalized filenames from CRM:")
    crm_filenames = get_crm_normalized_filenames(10)
    for i, fn in enumerate(crm_filenames, 1):
        print(f"CRM {i}: {repr(fn)}")
    print("\n[COMPARE] First 10 normalized filenames from DB:")
    db_filenames = get_db_normalized_filenames(10)
    for i, fn in enumerate(db_filenames, 1):
        print(f"DB  {i}: {repr(fn)}")

if __name__ == "__main__":
    main() 