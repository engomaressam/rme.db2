from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "PMO@1234"
}

# Directory containing PDFs
CVS_DIRECTORY = r"C:\cvs"

def get_db_connection():
    """Create a database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

@app.route('/api/search', methods=['GET'])
def search_cvs():
    """Search CVs by keywords in OCR text and other fields"""
    try:
        # Get search parameters
        keywords = request.args.get('keywords', '').strip()
        department = request.args.get('department', '').strip()
        job_title = request.args.get('job_title', '').strip()

        conditions = []
        params = []

        # Add keyword search if provided
        if keywords:
            keyword_list = [keyword.strip() for keyword in keywords.split(',')]
            keyword_conditions = " OR ".join([
                "LOWER(ocr_result) LIKE LOWER(%s)"
                for _ in keyword_list
            ])
            conditions.append(f"({keyword_conditions})")
            params.extend([f"%{keyword}%" for keyword in keyword_list])

        # Add department filter if provided
        if department:
            conditions.append("LOWER(department) = LOWER(%s)")
            params.append(department)

        # Add job title filter if provided
        if job_title:
            conditions.append("LOWER(job_title) = LOWER(%s)")
            params.append(job_title)

        # Combine all conditions with AND
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Connect to database and execute search
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                    SELECT id, pdf_filename, name, email, department, job_title, 
                           years_of_experience, current_company, location
                    FROM pdf_extracted_data
                    WHERE {where_clause}
                """
                cur.execute(query, params)
                results = cur.fetchall()

        # Convert results to list of dictionaries
        cvs = [dict(row) for row in results]
        return jsonify({
            "count": len(cvs),
            "results": cvs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cv/<filename>', methods=['GET'])
def get_cv_file(filename):
    """Serve CV PDF file"""
    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)
        file_path = os.path.join(CVS_DIRECTORY, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/departments', methods=['GET'])
def get_departments():
    """Get list of all departments"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT department 
                    FROM pdf_extracted_data 
                    WHERE department IS NOT NULL 
                    ORDER BY department
                """)
                results = cur.fetchall()
                departments = [row['department'] for row in results]
                return jsonify(departments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/job-titles', methods=['GET'])
def get_job_titles():
    """Get list of all job titles"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT job_title 
                    FROM pdf_extracted_data 
                    WHERE job_title IS NOT NULL 
                    ORDER BY job_title
                """)
                results = cur.fetchall()
                job_titles = [row['job_title'] for row in results]
                return jsonify(job_titles)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cv/details/<int:cv_id>', methods=['GET'])
def get_cv_details(cv_id):
    """Get detailed information about a specific CV"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, pdf_filename, name, email, department, job_title,
                           years_of_experience, current_company, location, languages,
                           certifications, project_types, skills, graduation_year,
                           university, linkedin
                    FROM pdf_extracted_data
                    WHERE id = %s
                """, (cv_id,))
                result = cur.fetchone()

                if not result:
                    return jsonify({"error": "CV not found"}), 404

                return jsonify(dict(result))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 