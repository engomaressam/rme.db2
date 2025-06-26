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

def add_summary_column():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'pdf_extracted_data' 
                        AND column_name = 'summary'
                    ) THEN
                        ALTER TABLE pdf_extracted_data ADD COLUMN summary TEXT;
                    END IF;
                END $$;
            """)
            conn.commit()
    print("Added summary column if missing.")

def get_first_20_percent(text):
    if not text:
        return ""
    words = text.split()
    twenty_percent_length = max(1, int(len(words) * 0.2))
    return " ".join(words[:twenty_percent_length])

def generate_summary(text):
    try:
        # Ensure text is not empty before calling API
        if not text.strip():
            print("Warning: Empty text provided for summary generation.")
            return None

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes CV content. Focus on the candidate's key qualifications, experience, and career objectives. Keep the summary concise and professional."},
                {"role": "user", "content": f"Please provide a concise summary of this CV content:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def update_summaries():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get rows where summary is NULL
    cur.execute("""
        SELECT id, ocr_result 
        FROM pdf_extracted_data 
        WHERE summary IS NULL
    """)
    
    rows = cur.fetchall()
    print(f"Processing {len(rows)} rows...")
    
    for row_id, extracted_text in rows:
        if not extracted_text:
            print(f"Skipping row {row_id} - no extracted text")
            continue
            
        # Get first 20% of text
        first_20_percent = get_first_20_percent(extracted_text)
        print(f"\nProcessing row {row_id}")
        print(f"First 20% text length: {len(first_20_percent)} characters")
        
        # Generate summary
        summary = generate_summary(first_20_percent)
        if summary:
            # Update database
            cur.execute("""
                UPDATE pdf_extracted_data 
                SET summary = %s 
                WHERE id = %s
            """, (summary, row_id))
            print(f"Updated summary for row {row_id}")
            print(f"Summary: {summary[:100]}...")  # Print first 100 chars of summary
        else:
            print(f"Failed to generate summary for row {row_id}")
    
    conn.commit()
    cur.close()
    conn.close()
    print("\nFinished processing rows")

def main():
    add_summary_column()
    update_summaries() # Process all rows where summary is NULL

if __name__ == "__main__":
    main() 