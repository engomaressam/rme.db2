import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from requests_ntlm import HttpNtlmAuth
from datetime import datetime
import csv
import pandas as pd
import random

# CRM API Configuration
CRM_URL = "https://rmecrm.rowad-rme.com/RMECRM/api/data/v8.2"
USERNAME = "Rowad\\Omar Essam"
PASSWORD = "PMO@1234"

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

def get_session():
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    })
    return session

def add_new_crm_columns():
    columns = [
        ("crm_new_jauid", "INTEGER"),
        ("crm_modifiedby_value", "VARCHAR(100)"),
        ("crm_modifiedbyname", "VARCHAR(255)"),
        ("crm_new_jobofferstatus", "VARCHAR(100)")
    ]
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for col, coltype in columns:
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'pdf_extracted_data' 
                            AND column_name = '{col}'
                        ) THEN
                            ALTER TABLE pdf_extracted_data ADD COLUMN {col} {coltype};
                        END IF;
                    END $$;
                """)
            conn.commit()
    print("Added new CRM columns if missing.")

def fetch_crm_applications_with_annotations(session):
    fields = [
        "new_jobapplicationid", "new_fullname", "new_contactphone", "new_telephonenumber",
        "new_jauid", "modifiedon", "new_jobofferstatus",
        "createdon"
    ]
    url = f"{CRM_URL}/new_jobapplications?$select={','.join(fields)}&$top=5000"
    response = session.get(url)
    response.raise_for_status()
    applications = response.json().get("value", [])
    crm_data = []
    for app in applications:
        app_id = app.get("new_jobapplicationid")
        createdon = app.get("createdon")
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
                    'jauid': app.get("new_jauid"),
                    'jobofferstatus': app.get("new_jobofferstatus")
                })
    return crm_data

def update_database(crm_data):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for item in crm_data:
                normalized_filename = item['normalized_filename']
                jauid = item['jauid']
                jobofferstatus = item['jobofferstatus']
                print(f"Updating: normalized_filename={normalized_filename}, jauid={jauid}, jobofferstatus={jobofferstatus}")
                cur.execute("""
                    UPDATE pdf_extracted_data SET
                        crm_new_jauid = %s,
                        crm_new_jobofferstatus = %s
                    WHERE LOWER(REPLACE(pdf_filename, ' ', '')) = %s
                """, (jauid, jobofferstatus, normalized_filename))
            conn.commit()
    print("Database updated with new CRM fields.")

def print_sample_filenames(crm_data):
    print("Sample normalized_filename values from CRM (repr):")
    for item in crm_data[:5]:
        print(repr(item['normalized_filename']))
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pdf_filename FROM pdf_extracted_data LIMIT 5")
            rows = cur.fetchall()
            print("Sample pdf_filename values from DB (repr):")
            for row in rows:
                print(repr(row[0]))

def write_sample_filenames_to_file(crm_data):
    with open('filename_samples.txt', 'w', encoding='utf-8') as f:
        f.write("Sample normalized_filename values from CRM (repr):\n")
        for item in crm_data[:5]:
            f.write(repr(item['normalized_filename']) + '\n')
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pdf_filename FROM pdf_extracted_data LIMIT 5")
                rows = cur.fetchall()
                f.write("Sample pdf_filename values from DB (repr):\n")
                for row in rows:
                    f.write(repr(row[0]) + '\n')

def write_sample_annotations_to_file(session):
    url = f"{CRM_URL}/annotations?$select=filename,_objectid_value,createdon&$top=5"
    response = session.get(url)
    response.raise_for_status()
    annotations = response.json().get('value', [])
    with open('annotation_samples.txt', 'w', encoding='utf-8') as f:
        f.write("Sample annotation records from CRM (filename, _objectid_value, createdon):\n")
        for ann in annotations:
            f.write(f"filename: {repr(ann.get('filename'))}, _objectid_value: {repr(ann.get('_objectid_value'))}, createdon: {repr(ann.get('createdon'))}\n")

def export_crm_2025_jauid_map(session):
    fields = [
        "new_jauid", "new_contactphone", "new_telephonenumber", "createdon"
    ]
    # Filter for 2025 createdon
    url = f"{CRM_URL}/new_jobapplications?$select={','.join(fields)}&$filter=createdon ge 2025-01-01T00:00:00Z&$top=5000"
    response = session.get(url)
    response.raise_for_status()
    applications = response.json().get("value", [])
    with open('crm_2025_jauid_map.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for app in applications:
            writer.writerow({field: app.get(field, '') for field in fields})

def check_phones_in_db():
    df = pd.read_csv('crm_2025_jauid_map.csv')
    n = len(df)
    # Get 10 random indices from the middle 60% of the file
    start = n // 5
    end = n - n // 5
    indices = random.sample(range(start, end), 10)
    phones = set(df.iloc[i]['new_contactphone'] for i in indices) | set(df.iloc[i]['new_telephonenumber'] for i in indices)
    phones = {str(p) for p in phones if pd.notnull(p)}
    results = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for phone in phones:
                cur.execute("""
                    SELECT COUNT(*) FROM pdf_extracted_data
                    WHERE crm_contactphone = %s OR crm_telephonenumber = %s
                """, (phone, phone))
                count = cur.fetchone()[0]
                results.append(f"Phone: {phone} | Matches in DB: {count}")
    with open('phone_check_results.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

def check_emails_in_db():
    df = pd.read_csv('crm_2025_jauid_map.csv')
    if 'crm_email' not in df.columns:
        print('No crm_email column in CRM export.')
        return
    n = len(df)
    start = n // 5
    end = n - n // 5
    indices = random.sample(range(start, end), 10)
    emails = set(str(df.iloc[i]['crm_email']) for i in indices if pd.notnull(df.iloc[i]['crm_email']))
    results = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for email in emails:
                cur.execute("""
                    SELECT COUNT(*) FROM pdf_extracted_data
                    WHERE crm_email = %s
                """, (email,))
                count = cur.fetchone()[0]
                results.append(f"Email: {email} | Matches in DB: {count}")
    with open('email_check_results.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

def print_all_column_names():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'pdf_extracted_data'")
            columns = [row[0] for row in cur.fetchall()]
            print('Current columns in pdf_extracted_data:')
            for col in columns:
                print(col)

def main():
    print_all_column_names()
    add_new_crm_columns()
    print("Checking if CRM phone numbers exist in DB...")
    check_phones_in_db()
    print("Checking if CRM emails exist in DB...")
    check_emails_in_db()
    print("Wrote phone and email check results to phone_check_results.txt and email_check_results.txt. Exiting early for review.")
    return
    # update_database(crm_data)
    # print("Done.")

if __name__ == "__main__":
    main() 