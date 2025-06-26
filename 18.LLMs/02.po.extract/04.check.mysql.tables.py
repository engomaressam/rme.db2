import mysql.connector
from mysql.connector import Error
import pandas as pd
from tabulate import tabulate
import sys
import os

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def connect_to_database():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        sys.exit(1)

def get_table_summary(connection):
    """Get a summary of all tables in the database"""
    cursor = connection.cursor()
    
    try:
        # Get list of tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        table_data = []
        for table in tables:
            table_name = table[0]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
            
            # Get creation time if available
            try:
                cursor.execute(f"""
                SELECT CREATE_TIME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = '{DB_CONFIG['database']}' 
                AND TABLE_NAME = '{table_name}'
                """)
                create_time = cursor.fetchone()[0]
            except:
                create_time = "Unknown"
            
            table_data.append([table_name, row_count, create_time])
        
        print("\nDatabase Tables Summary:")
        print(tabulate(table_data, headers=["Table Name", "Row Count", "Creation Time"]))
        
    except Error as e:
        print(f"Error getting table summary: {e}")
    finally:
        cursor.close()

def get_pdf_stats(connection):
    """Get statistics about the PDF files processed"""
    cursor = connection.cursor()
    
    try:
        # Check if po.pdfs table exists
        cursor.execute("SHOW TABLES LIKE 'po.pdfs'")
        if not cursor.fetchone():
            print("Table 'po.pdfs' does not exist")
            return
        
        # Get total PDFs count
        cursor.execute("SELECT COUNT(*) FROM `po.pdfs`")
        total_pdfs = cursor.fetchone()[0]
        
        # Get last processed PDF
        cursor.execute("""
        SELECT pdf_filename, processed_timestamp 
        FROM `po.pdfs` 
        ORDER BY processed_timestamp DESC 
        LIMIT 1
        """)
        last_pdf = cursor.fetchone()
        
        # Get top 5 folders with most PDFs
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
        
        print(f"\nPDF Processing Statistics:")
        print(f"Total PDFs processed: {total_pdfs}")
        if last_pdf:
            print(f"Last processed PDF: {last_pdf[0]} at {last_pdf[1]}")
        
        print("\nTop 5 Folders:")
        print(tabulate(top_folders, headers=["Folder", "PDF Count"]))
        
        # Get chunk statistics
        if total_pdfs > 0:
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
            
            print("\nText Chunk Statistics:")
            print(f"Total chunks extracted: {chunk_stats[0]}")
            print(f"Average chunks per PDF: {chunk_stats[1]}")
            print(f"Min chunks in a PDF: {chunk_stats[2]}")
            print(f"Max chunks in a PDF: {chunk_stats[3]}")
            
            # Get sample PDFs with different chunk counts
            print("\nSample PDFs:")
            
            # PDF with most chunks
            cursor.execute("""
            SELECT p.pdf_filename, COUNT(c.id) as chunk_count
            FROM `po.pdfs` p
            JOIN `po.pdf_chunks` c ON p.id = c.pdf_id
            GROUP BY p.id
            ORDER BY chunk_count DESC
            LIMIT 1
            """)
            max_chunks_pdf = cursor.fetchone()
            if max_chunks_pdf:
                print(f"PDF with most chunks: {max_chunks_pdf[0]} ({max_chunks_pdf[1]} chunks)")
            
            # PDF with fewest chunks
            cursor.execute("""
            SELECT p.pdf_filename, COUNT(c.id) as chunk_count
            FROM `po.pdfs` p
            JOIN `po.pdf_chunks` c ON p.id = c.pdf_id
            GROUP BY p.id
            ORDER BY chunk_count ASC
            LIMIT 1
            """)
            min_chunks_pdf = cursor.fetchone()
            if min_chunks_pdf:
                print(f"PDF with fewest chunks: {min_chunks_pdf[0]} ({min_chunks_pdf[1]} chunks)")
            
            # Get most recent 3 PDFs processed
            cursor.execute("""
            SELECT pdf_filename, processed_timestamp 
            FROM `po.pdfs` 
            ORDER BY processed_timestamp DESC 
            LIMIT 3
            """)
            recent_pdfs = cursor.fetchall()
            
            print("\nMost Recently Processed PDFs:")
            print(tabulate(recent_pdfs, headers=["Filename", "Processed At"]))
            
    except Error as e:
        print(f"Error getting PDF statistics: {e}")
    finally:
        cursor.close()

def main():
    """Main function"""
    connection = connect_to_database()
    
    if connection.is_connected():
        print("=" * 80)
        print("MYSQL DATABASE STATUS REPORT")
        print("=" * 80)
        
        # Get table summary
        get_table_summary(connection)
        
        # Get PDF statistics
        get_pdf_stats(connection)
        
        # Close connection
        connection.close()
        print("\nConnection closed")

if __name__ == "__main__":
    main()
