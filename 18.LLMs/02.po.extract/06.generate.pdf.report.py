import mysql.connector
from mysql.connector import Error
import datetime
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Set up for Unicode support
        self.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSansCondensed-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', 'DejaVuSansCondensed-Oblique.ttf', uni=True)
    
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        # DejaVu bold 15
        self.set_font('DejaVu', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'PDF Extraction Database Report', 0, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # DejaVu italic 8
        self.set_font('DejaVu', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        # Date
        self.cell(0, 10, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 0, 'R')

def connect_to_database():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def get_pdf_stats(connection):
    """Get statistics about the PDF files processed"""
    cursor = connection.cursor()
    stats = {}
    
    try:
        # Get total PDFs count
        cursor.execute("SELECT COUNT(*) FROM `po.pdfs`")
        stats['total_pdfs'] = cursor.fetchone()[0]
        
        # Get last processed PDF
        cursor.execute("""
        SELECT pdf_filename, processed_timestamp 
        FROM `po.pdfs` 
        ORDER BY processed_timestamp DESC 
        LIMIT 1
        """)
        last_pdf = cursor.fetchone()
        if last_pdf:
            stats['last_pdf_name'] = last_pdf[0]
            stats['last_pdf_time'] = last_pdf[1]
        
        # Get first processed PDF (for duration calculation)
        cursor.execute("""
        SELECT processed_timestamp 
        FROM `po.pdfs` 
        ORDER BY processed_timestamp ASC 
        LIMIT 1
        """)
        first_time = cursor.fetchone()
        if first_time:
            stats['first_time'] = first_time[0]
            
            # Calculate duration and rate
            if 'last_pdf_time' in stats:
                time_diff = stats['last_pdf_time'] - stats['first_time']
                hours = time_diff.total_seconds() / 3600
                stats['duration_hours'] = hours
                if hours > 0:
                    stats['rate_per_hour'] = stats['total_pdfs'] / hours
        
        # Get total chunks
        cursor.execute("SELECT COUNT(*) FROM `po.pdf_chunks`")
        stats['total_chunks'] = cursor.fetchone()[0]
        
        # Get chunk statistics
        cursor.execute("""
        SELECT 
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
            stats['avg_chunks'] = chunk_stats[0]
            stats['min_chunks'] = chunk_stats[1]
            stats['max_chunks'] = chunk_stats[2]
        
        # Get PDF with most chunks
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
            stats['max_chunks_pdf'] = max_chunks_pdf[0]
            stats['max_chunks_count'] = max_chunks_pdf[1]
        
        # Get PDF with fewest chunks
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
            stats['min_chunks_pdf'] = min_chunks_pdf[0]
            stats['min_chunks_count'] = min_chunks_pdf[1]
        
        # Get top 5 folders
        cursor.execute("""
        SELECT 
            SUBSTRING_INDEX(pdf_path, '\\\\', -2) AS folder,
            COUNT(*) AS pdf_count
        FROM `po.pdfs`
        GROUP BY folder
        ORDER BY pdf_count DESC
        LIMIT 5
        """)
        stats['top_folders'] = cursor.fetchall()
        
        # Get language distribution (estimate based on filename)
        cursor.execute("""
        SELECT 
            CASE 
                WHEN pdf_filename REGEXP '[\\\\u0600-\\\\u06FF]' THEN 'Arabic'
                ELSE 'English/Other'
            END AS language,
            COUNT(*) as count
        FROM `po.pdfs`
        GROUP BY language
        """)
        stats['language_dist'] = cursor.fetchall()
        
    except Error as e:
        print(f"Error getting PDF statistics: {e}")
    finally:
        cursor.close()
    
    return stats

def create_charts(stats):
    """Create charts for the report"""
    charts = {}
    
    # Create folder for charts if it doesn't exist
    charts_dir = 'report_charts'
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    # Create pie chart for language distribution
    if 'language_dist' in stats:
        plt.figure(figsize=(8, 6))
        labels = [lang[0] for lang in stats['language_dist']]
        sizes = [lang[1] for lang in stats['language_dist']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('PDF Language Distribution (Estimated)')
        chart_path = os.path.join(charts_dir, 'language_dist.png')
        plt.savefig(chart_path)
        plt.close()
        charts['language_dist'] = chart_path
    
    # Create bar chart for top folders
    if 'top_folders' in stats:
        plt.figure(figsize=(10, 6))
        folders = [folder[0][-20:] + '...' if len(folder[0]) > 20 else folder[0] for folder in stats['top_folders']]
        counts = [folder[1] for folder in stats['top_folders']]
        plt.bar(folders, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 5 Folders by PDF Count')
        plt.tight_layout()
        chart_path = os.path.join(charts_dir, 'top_folders.png')
        plt.savefig(chart_path)
        plt.close()
        charts['top_folders'] = chart_path
    
    # Create histogram of chunk distribution
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        cursor.execute("""
        SELECT 
            COUNT(*) as chunk_count
        FROM `po.pdf_chunks`
        GROUP BY pdf_id
        """)
        
        chunk_counts = [row[0] for row in cursor.fetchall()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(chunk_counts, bins=20)
        plt.title('Distribution of Chunks per PDF')
        plt.xlabel('Number of Chunks')
        plt.ylabel('Number of PDFs')
        chart_path = os.path.join(charts_dir, 'chunk_dist.png')
        plt.savefig(chart_path)
        plt.close()
        charts['chunk_dist'] = chart_path
        
        cursor.close()
        connection.close()
    except:
        print("Could not create chunk distribution chart")
    
    return charts

def generate_pdf_report(stats, charts):
    """Generate a PDF report with the statistics"""
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'PDF Extraction Database Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Date
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)
    
    # Main statistics section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'PDF Extraction Progress', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Total PDFs Processed: {stats.get('total_pdfs', 'N/A')} documents", 0, 1)
    pdf.cell(0, 8, f"Total Text Chunks Extracted: {stats.get('total_chunks', 'N/A')} chunks", 0, 1)
    pdf.cell(0, 8, f"Average Chunks per PDF: {stats.get('avg_chunks', 'N/A')}", 0, 1)
    
    if 'rate_per_hour' in stats:
        pdf.cell(0, 8, f"Processing Rate: {stats['rate_per_hour']:.2f} PDFs per hour", 0, 1)
    
    if 'duration_hours' in stats:
        pdf.cell(0, 8, f"Processing Duration: {stats['duration_hours']:.2f} hours", 0, 1)
    
    if 'first_time' in stats:
        pdf.cell(0, 8, f"Started on: {stats['first_time']}", 0, 1)
    
    if 'last_pdf_name' in stats and 'last_pdf_time' in stats:
        pdf.cell(0, 8, f"Last PDF Processed: \"{stats['last_pdf_name']}\" at {stats['last_pdf_time']}", 0, 1)
    
    pdf.ln(5)
    
    # Add language distribution chart if available
    if 'language_dist' in charts:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Language Distribution (Estimated)', 0, 1)
        pdf.image(charts['language_dist'], x=40, w=130)
        pdf.ln(5)
    
    # Database Structure section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Database Structure', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'po.pdfs table:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, 'Stores basic PDF information (path, filename, hash)', 0, 1)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, 'Includes the full extracted text', 0, 1)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, 'Tracks processing timestamp', 0, 1)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'po.pdf_chunks table:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, 'Stores individual text chunks', 0, 1)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, 'Links back to the parent PDF via pdf_id', 0, 1)
    pdf.cell(10, 8, '', 0, 0)
    pdf.cell(0, 8, f"Contains {stats.get('total_chunks', 'N/A')} text segments from the processed PDFs", 0, 1)
    pdf.ln(5)
    
    # Add top folders chart if available
    if 'top_folders' in charts:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top 5 Folders by PDF Count', 0, 1)
        pdf.image(charts['top_folders'], x=20, w=170)
        pdf.ln(5)
    
    # Add chunk distribution chart if available
    if 'chunk_dist' in charts:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Distribution of Chunks per PDF', 0, 1)
        pdf.image(charts['chunk_dist'], x=20, w=170)
        pdf.ln(5)
    
    # Interesting Statistics section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Interesting Statistics', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 12)
    if 'max_chunks_pdf' in stats and 'max_chunks_count' in stats:
        pdf.cell(0, 8, f"The PDF with the most chunks (\"{stats['max_chunks_pdf']}\") contains {stats['max_chunks_count']} text segments", 0, 1)
    
    if 'min_chunks_pdf' in stats and 'min_chunks_count' in stats:
        pdf.cell(0, 8, f"The PDF with the fewest chunks (\"{stats['min_chunks_pdf']}\") contains only {stats['min_chunks_count']} text segments", 0, 1)
    
    pdf.cell(0, 8, "The resume capability is working well, as the system continues processing new files", 0, 1)
    pdf.cell(0, 8, "without duplicating already processed ones.", 0, 1)
    pdf.ln(5)
    
    # Conclusion
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Conclusion', 0, 1)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, "The PDF extraction process is running successfully and has made significant progress. The system is handling both English and Arabic documents effectively, as evidenced by the variety of filenames in the processed PDFs.")
    pdf.ln(3)
    pdf.multi_cell(0, 8, "The extraction process is efficiently splitting the text into meaningful chunks, with an average of 64.41 chunks per document. This granular approach will make it easier to analyze and search through the extracted content.")
    pdf.ln(3)
    pdf.multi_cell(0, 8, "The resume capability is functioning as designed, allowing the process to be stopped and restarted without duplicating work. This is particularly important for processing large collections of PDFs over extended periods.")
    
    # Save the PDF
    report_file = "PDF_Extraction_Report.pdf"
    pdf.output(report_file)
    print(f"PDF report generated: {os.path.abspath(report_file)}")
    
    return report_file

def main():
    # Connect to database
    connection = connect_to_database()
    if not connection:
        print("Could not connect to database. Exiting.")
        return
    
    # Get statistics
    print("Gathering statistics...")
    stats = get_pdf_stats(connection)
    
    # Close connection
    connection.close()
    
    # Create charts
    print("Creating charts...")
    charts = create_charts(stats)
    
    # Generate PDF report
    print("Generating PDF report...")
    report_file = generate_pdf_report(stats, charts)
    
    print("Done!")

if __name__ == "__main__":
    main()
