import os
import requests
from requests_ntlm import HttpNtlmAuth
from urllib.parse import quote
from datetime import datetime
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

# List of CRM fields to extract and update (add more as needed)
CRM_FIELDS = [
    "new_jobapplicationid",
    "new_fullname",
    "new_contactphone",
    "new_telephonenumber",
    "new_jauid",
    "new_jobofferstatus",
    "new_gender",
    "new_position",
    "new_employmenttype",
    "new_expectedsalary",
    "new_dateavailableforemployment",
    "new_currentsalary",
    "new_company",
    "new_graduationyear",
    "new_qualitiesattributes",
    "new_careergoals",
    "new_additionalinformation",
    "new_appstatus",
    "new_hrinterviewstatus",
    "new_technicalrating",
    "new_technicalinterviewcomments",
    "new_hrcomment",
    "createdon",
    "modifiedon",
    "new_howdidyouhearaboutrowad",
    "new_listouttheextrasocialactivities",
    "new_pleasesepcify"
]

# Map CRM fields to DB columns (prefix with crm_)
CRM_TO_DB = {field: f"crm_{field[4:]}" if field.startswith("new_") else f"crm_{field}" for field in CRM_FIELDS}

# Add columns to DB if missing
def add_crm_columns():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for crm_field, db_col in CRM_TO_DB.items():
                # Use TEXT for simplicity, can be refined
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'pdf_extracted_data' 
                            AND column_name = '{db_col}'
                        ) THEN
                            ALTER TABLE pdf_extracted_data ADD COLUMN {db_col} TEXT;
                        END IF;
                    END $$;
                """)
            conn.commit()
    print("[CRM] Ensured all extra CRM columns exist in the database.")

# Fetch CRM data and annotation filenames
def fetch_crm_data():
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        "Accept": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    })
    # Only fetch job applications created in 2025
    filter_condition = "createdon ge 2025-01-01T00:00:00Z and createdon le 2025-12-31T23:59:59Z"
    url = f"{CRM_URL}/new_jobapplications?$select={','.join(CRM_FIELDS)}&$filter={filter_condition}&$top=5000"
    print(f"[CRM] Fetching job applications for 2025 with fields: {', '.join(CRM_FIELDS)}")
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
                    **app
                })
    print(f"[CRM] Fetched {len(crm_data)} CRM records with annotation filenames.")
    return crm_data

# Update DB with CRM fields by normalized filename
def update_db_with_crm_fields(crm_data):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            updated = 0
            for item in crm_data:
                normalized_filename = item['normalized_filename']
                set_clauses = []
                values = []
                for crm_field, db_col in CRM_TO_DB.items():
                    value = item.get(crm_field)
                    set_clauses.append(f"{db_col} = %s")
                    values.append(value)
                set_clause = ", ".join(set_clauses)
                values.append(normalized_filename)
                cur.execute(f"""
                    UPDATE pdf_extracted_data SET {set_clause}
                    WHERE LOWER(REPLACE(pdf_filename, ' ', '')) = %s
                """, values)
                if cur.rowcount > 0:
                    updated += cur.rowcount
            conn.commit()
    print(f"[CRM] Updated {updated} rows in pdf_extracted_data with extra CRM fields.")

if __name__ == "__main__":
    print("[CRM] Adding extra CRM columns if missing...")
    add_crm_columns()
    print("[CRM] Fetching CRM data and annotation filenames...")
    crm_data = fetch_crm_data()
    print("[CRM] Updating database with CRM fields...")
    update_db_with_crm_fields(crm_data)
    print("[CRM] Done.") 