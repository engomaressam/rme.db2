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

def get_pdf_filenames_from_db():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pdf_filename FROM pdf_extracted_data")
            return [row[0] for row in cur.fetchall()]

def get_crm_data_with_annotation_filenames():
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    })
    # Get all job applications with their IDs, createdon, and new_jauid
    url = f"{CRM_URL}/new_jobapplications?$select=new_jobapplicationid,createdon,new_jauid&$top=5000"
    response = session.get(url)
    response.raise_for_status()
    applications = response.json().get('value', [])
    crm_data = []
    for app in applications:
        app_id = app.get('new_jobapplicationid')
        createdon = app.get('createdon')
        jauid = app.get('new_jauid')
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
                crm_data.append({
                    'normalized_filename': normalized,
                    'jauid': jauid
                })
    return crm_data

def update_db_with_jauid(crm_data):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for item in crm_data:
                normalized_filename = item['normalized_filename']
                jauid = item['jauid']
                cur.execute("""
                    UPDATE pdf_extracted_data 
                    SET crm_new_jauid = %s 
                    WHERE LOWER(REPLACE(pdf_filename, ' ', '')) = %s
                """, (jauid, normalized_filename))
            conn.commit()

def main():
    print("Fetching data from CRM (with annotation filenames)...")
    crm_data = get_crm_data_with_annotation_filenames()
    print(f"Found {len(crm_data)} annotation-based filenames in CRM")
    
    # Print first 5 CRM records
    print("\nSample CRM records:")
    for i, item in enumerate(crm_data[:5], 1):
        print(f"{i}. Normalized filename: {item['normalized_filename']}")
        print(f"   JAUID: {item['jauid']}")
    
    # Get and print first 5 database filenames
    print("\nSample database filenames:")
    db_filenames = get_pdf_filenames_from_db()
    for i, filename in enumerate(db_filenames[:5], 1):
        normalized = filename.replace(' ', '').lower()
        print(f"{i}. Original: {filename}")
        print(f"   Normalized: {normalized}")
    
    print("\nUpdating database with new_jauid...")
    update_db_with_jauid(crm_data)
    print("Database update complete")

if __name__ == "__main__":
    main() 