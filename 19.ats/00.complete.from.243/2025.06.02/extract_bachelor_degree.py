import os
import psycopg2
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

def add_bachelor_degree_column():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'pdf_extracted_data' AND column_name = 'bachelor_degree'
            """)
            if not cur.fetchone():
                cur.execute("ALTER TABLE pdf_extracted_data ADD COLUMN bachelor_degree TEXT")
                print("Added bachelor_degree column.")
            else:
                print("bachelor_degree column already exists.")
            conn.commit()

def extract_bachelor_degree_with_gpt(ocr_text):
    try:
        text_length = len(ocr_text)
        first_portion = ocr_text[:int(text_length * 0.3)]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Extract the type of bachelor degree (e.g., Bachelor of Science in Civil Engineering, Bachelor of Arts in English Literature, etc.) from the following CV text. If not found, return 'Not specified'. Return ONLY the degree name as a string."
            }, {
                "role": "user",
                "content": first_portion
            }],
            temperature=0.1,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT extraction error: {e}")
        return None

def process_bachelor_degrees():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ocr_result FROM pdf_extracted_data 
                WHERE ocr_result IS NOT NULL AND (bachelor_degree IS NULL OR bachelor_degree = '')
            """)
            records = cur.fetchall()
            print(f"Found {len(records)} records to process.")
            processed = 0
            failed = 0
            for record_id, ocr_text in records:
                print(f"Processing CV ID: {record_id}")
                degree = extract_bachelor_degree_with_gpt(ocr_text)
                if not degree:
                    print(f"Failed to extract bachelor degree for CV ID: {record_id}")
                    failed += 1
                    continue
                try:
                    cur.execute("""
                        UPDATE pdf_extracted_data SET bachelor_degree = %s WHERE id = %s
                    """, (degree, record_id))
                    conn.commit()
                    processed += 1
                    print(f"✓ Extracted: {degree}")
                except Exception as e:
                    print(f"Database update error for CV ID {record_id}: {e}")
                    conn.rollback()
                    failed += 1
            print(f"\nProcessing completed:")
            print(f"✓ Successfully processed: {processed} records")
            print(f"❌ Failed to process: {failed} records")

if __name__ == "__main__":
    add_bachelor_degree_column()
    process_bachelor_degrees() 