import mysql.connector
from mysql.connector import Error
import os
import datetime

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def generate_report():
    """Generate a report on the MySQL database tables"""
    try:
        # Connect to the database
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # Create report file
        report_file = "mysql_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"PDF EXTRACTION DATABASE REPORT - {datetime.datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Focus only on our PDF tables
            tables = ["po.pdfs", "po.pdf_chunks"]
            
            f.write(f"Checking PDF extraction tables in database {DB_CONFIG['database']}:\n\n")
            
            # Process each table
            for table_name in tables:
                try:
                    # Check if table exists
                    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                    if not cursor.fetchone():
                        f.write(f"TABLE: {table_name} - DOES NOT EXIST\n")
                        f.write("-" * 80 + "\n\n")
                        continue
                    
                    f.write(f"TABLE: {table_name}\n")
                    f.write("-" * 80 + "\n")
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                    row_count = cursor.fetchone()[0]
                    f.write(f"Row count: {row_count}\n")
                    
                    # Get table structure
                    cursor.execute(f"DESCRIBE `{table_name}`")
                    columns = cursor.fetchall()
                    f.write("\nColumns:\n")
                    for col in columns:
                        f.write(f"  - {col[0]}: {col[1]} {col[2]} {col[3]}\n")
                except Error as e:
                    f.write(f"Error accessing table {table_name}: {str(e)}\n\n")
                    continue
                
                # Get specific stats for our tables
                if table_name == "po.pdfs":
                    f.write("\nPDF Statistics:\n")
                    
                    # Last processed PDF
                    cursor.execute("""
                    SELECT pdf_filename, processed_timestamp 
                    FROM `po.pdfs` 
                    ORDER BY processed_timestamp DESC 
                    LIMIT 1
                    """)
                    last_pdf = cursor.fetchone()
                    if last_pdf:
                        f.write(f"  - Last processed PDF: {last_pdf[0]} at {last_pdf[1]}\n")
                    
                    # Top folders
                    cursor.execute("""
                    SELECT 
                        SUBSTRING_INDEX(pdf_path, '\\\\', -2) AS folder,
                        COUNT(*) AS pdf_count
                    FROM `po.pdfs`
                    GROUP BY folder
                    ORDER BY pdf_count DESC
                    LIMIT 5
                    """)
                    top_folders = cursor.fetchall()
                    f.write("\n  Top 5 folders:\n")
                    for folder in top_folders:
                        f.write(f"    - {folder[0]}: {folder[1]} PDFs\n")
                
                elif table_name == "po.pdf_chunks":
                    f.write("\nChunk Statistics:\n")
                    
                    # Chunk counts
                    cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        ROUND(AVG(chunk_count), 2) AS avg_chunks,
                        MIN(chunk_count) AS min_chunks,
                        MAX(chunk_count) AS max_chunks
                    FROM (
                        SELECT 
                            pdf_id, 
                            COUNT(*) AS chunk_count
                        FROM `po.pdf_chunks`
                        GROUP BY pdf_id
                    ) AS chunk_counts
                    """)
                    chunk_stats = cursor.fetchone()
                    if chunk_stats:
                        f.write(f"  - Total chunks: {chunk_stats[0]}\n")
                        f.write(f"  - Average chunks per PDF: {chunk_stats[1]}\n")
                        f.write(f"  - Min chunks in a PDF: {chunk_stats[2]}\n")
                        f.write(f"  - Max chunks in a PDF: {chunk_stats[3]}\n")
                
                f.write("\n\n")
            
            # Write summary
            f.write("SUMMARY:\n")
            f.write("-" * 80 + "\n")
            
            try:
                # Check if tables exist first
                cursor.execute("SHOW TABLES LIKE 'po.pdfs'")
                pdfs_exists = cursor.fetchone() is not None
                
                cursor.execute("SHOW TABLES LIKE 'po.pdf_chunks'")
                chunks_exists = cursor.fetchone() is not None
                
                if not pdfs_exists or not chunks_exists:
                    f.write("Cannot generate summary: One or more required tables do not exist.\n")
                else:
                    # Get total PDFs
                    cursor.execute("SELECT COUNT(*) FROM `po.pdfs`")
                    pdf_count = cursor.fetchone()[0]
                    
                    # Get total chunks
                    cursor.execute("SELECT COUNT(*) FROM `po.pdf_chunks`")
                    chunk_count = cursor.fetchone()[0]
                    
                    f.write(f"Total PDFs processed: {pdf_count}\n")
                    f.write(f"Total text chunks extracted: {chunk_count}\n")
                    
                    if pdf_count > 0:
                        f.write(f"Average chunks per PDF: {chunk_count / pdf_count:.2f}\n")
                    
                    # Get processing rate
                    cursor.execute("""
                    SELECT 
                        MIN(processed_timestamp) as first_pdf,
                        MAX(processed_timestamp) as last_pdf,
                        COUNT(*) as pdf_count
                    FROM `po.pdfs`
                    """)
                    time_stats = cursor.fetchone()
                    
                    if time_stats and time_stats[0] and time_stats[1]:
                        first_time = time_stats[0]
                        last_time = time_stats[1]
                        pdf_count = time_stats[2]
                        
                        time_diff = last_time - first_time
                        hours = time_diff.total_seconds() / 3600
                        
                        if hours > 0:
                            rate = pdf_count / hours
                            f.write(f"\nProcessing rate: {rate:.2f} PDFs per hour\n")
                            f.write(f"Processing started: {first_time}\n")
                            f.write(f"Last PDF processed: {last_time}\n")
                            f.write(f"Total processing time: {hours:.2f} hours\n")
            except Error as e:
                f.write(f"Error generating summary: {str(e)}\n")
        
        print(f"Report generated: {os.path.abspath(report_file)}")
        
        # Close connection
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_report()
