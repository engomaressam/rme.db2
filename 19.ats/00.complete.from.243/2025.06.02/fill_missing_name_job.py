import psycopg2
import os
from openai import OpenAI
import json
from dotenv import dotenv_values

# Load environment variables from .env file
config = dotenv_values(".env")

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

# Initialize OpenAI client
client = OpenAI(api_key=config.get('OPENAI_API_KEY'))

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_first_25_percent(text):
    if not text:
        return ""
    words = text.split()
    twenty_five_percent_length = max(1, int(len(words) * 0.25))
    return " ".join(words[:twenty_five_percent_length])

def extract_name_job_with_gpt(text):
    try:
        # Ensure text is not empty before calling API
        if not text.strip():
            print("Warning: Empty text provided for name/job extraction.")
            return None, None

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract the candidate's full name and current job title from the provided text. Provide the output as a JSON object with keys 'name' and 'current_job'. If a field is not found, use null. Only provide the JSON object."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.1
        )
        result_str = response.choices[0].message.content.strip()
        try:
            result = json.loads(result_str)
            name = result.get('name')
            current_job = result.get('current_job')
            return name, current_job
        except json.JSONDecodeError:
            print(f"Error decoding JSON from GPT response: {result_str}")
            return None, None
    except Exception as e:
        print(f"Error generating name/job: {e}")
        return None, None

def fill_missing_fields():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get rows where name or current_job is NULL
    cur.execute("""
        SELECT id, ocr_result 
        FROM pdf_extracted_data 
        WHERE name IS NULL OR current_job IS NULL
    """)
    
    rows = cur.fetchall()
    print(f"Found {len(rows)} rows with missing name or current job.")
    
    processed_count = 0
    updated_count = 0

    for row_id, ocr_text in rows:
        processed_count += 1
        print(f"\nProcessing row {row_id} ({processed_count}/{len(rows)})...")

        if not ocr_text:
            print(f"Skipping row {row_id} - no OCR text available.")
            continue
            
        # Get first 25% of text
        first_25_percent = get_first_25_percent(ocr_text)
        print(f"Using first {len(first_25_percent)} characters (approx. 25% of text) for GPT.")
        
        # Extract name and current job using GPT
        name, current_job = extract_name_job_with_gpt(first_25_percent)
        
        update_set_clauses = []
        update_values = []
        
        # Only update if the value was extracted and the DB field is NULL
        if name is not None:
             cur.execute("SELECT name FROM pdf_extracted_data WHERE id = %s", (row_id,))
             current_name = cur.fetchone()[0]
             if current_name is None:
                update_set_clauses.append("name = %s")
                update_values.append(name)
                print(f" - Extracted and will update name: {name}")
             else:
                 print(f" - Extracted name but DB field is not NULL: {current_name}")
        else:
             print(" - Name not extracted by GPT.")

        if current_job is not None:
             cur.execute("SELECT current_job FROM pdf_extracted_data WHERE id = %s", (row_id,))
             current_job_db = cur.fetchone()[0]
             if current_job_db is None:
                update_set_clauses.append("current_job = %s")
                update_values.append(current_job)
                print(f" - Extracted and will update current_job: {current_job}")
             else:
                 print(f" - Extracted current_job but DB field is not NULL: {current_job_db}")
        else:
             print(" - Current job not extracted by GPT.")

        if update_set_clauses:
            update_query = f"UPDATE pdf_extracted_data SET {', '.join(update_set_clauses)} WHERE id = %s"
            update_values.append(row_id)
            try:
                cur.execute(update_query, update_values)
                conn.commit()
                updated_count += 1
                print(f"Successfully updated row {row_id}.")
            except Exception as e:
                print(f"[ERROR] Database update failed for row {row_id}: {e}")
                conn.rollback()
        else:
            print(f"No fields to update for row {row_id}.")

    cur.close()
    conn.close()
    print(f"\nFinished processing. Total rows processed: {processed_count}, Rows updated: {updated_count}")

def main():
    fill_missing_fields()

if __name__ == "__main__":
    main() 