import psycopg2
import json

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

# Function to get database connection
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

EXTRACT_JSON = "extract 1.json"

NEW_COLUMNS = [
    ("crm_name2", "VARCHAR(255)"),
    ("crm_email2", "VARCHAR(255)"),
    ("crm_jauid2", "INTEGER")
]

def add_new_columns():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for col, coltype in NEW_COLUMNS:
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
    print("Added new columns if missing.")

def update_from_json():
    conn = get_db_connection()
    cur = conn.cursor()

    print(f"Reading data from {EXTRACT_JSON}...")
    try:
        with open(EXTRACT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully read {len(data)} records from {EXTRACT_JSON}.")
    except FileNotFoundError:
        print(f"Error: {EXTRACT_JSON} not found.")
        conn.close()
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {EXTRACT_JSON}.")
        conn.close()
        return
    except Exception as e:
        print(f"An error occurred while reading {EXTRACT_JSON}: {e}")
        conn.close()
        return

    update_count = 0
    print("Starting database update...")
    for record in data:
        job_application_id = record.get('new_jobapplicationid')
        new_name = record.get('new_name')
        new_email = record.get('new_email')
        new_jauid = record.get('new_jauid')

        if not job_application_id:
            print(f"Skipping record due to missing job_application_id: {record}")
            continue

        # Update query to use COALESCE to only fill nulls and use PostgreSQL placeholders (%s)
        sql = """
        UPDATE pdf_extracted_data
        SET
            crm_name2 = COALESCE(crm_name2, %s),
            crm_email2 = COALESCE(crm_email2, %s),
            crm_jauid2 = COALESCE(crm_jauid2, %s)
        WHERE crm_applicationid = %s;
        """
        try:
            cur.execute(sql, (new_name, new_email, new_jauid, job_application_id))
            if cur.rowcount > 0:
                update_count += cur.rowcount

        except psycopg2.Error as e:
            print(f"Database error updating record {job_application_id}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred updating record {job_application_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Database update finished. Total rows updated (filling nulls): {update_count}")

def main():
    add_new_columns()
    update_from_json()

if __name__ == "__main__":
    main() 