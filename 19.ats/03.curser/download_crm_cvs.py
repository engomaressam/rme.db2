import os
import requests
from requests_ntlm import HttpNtlmAuth
import base64
from datetime import datetime
from urllib.parse import urljoin, quote
import json

# CRM Configuration
CRM_BASE_URL = "https://rmecrm.rowad-rme.com/RMECRM/"
API_ENDPOINT = "api/data/v8.2/new_jobapplications"
USERNAME = "Rowad\\Omar Essam"
PASSWORD = "PMO@1234"

# Local directory to save CVs
CVS_DIRECTORY = r"C:\cvs"
os.makedirs(CVS_DIRECTORY, exist_ok=True)

def get_ntlm_session():
    """Create a session with NTLM authentication"""
    session = requests.Session()
    session.auth = HttpNtlmAuth(USERNAME, PASSWORD)
    session.headers.update({
        'Accept': 'application/json',
        'OData-MaxVersion': '4.0',
        'OData-Version': '4.0'
    })
    session.verify = False  # Only if needed for self-signed certificates
    return session

def get_annotations(session, next_link):
    """Fetch annotations using the nextLink URL"""
    try:
        response = session.get(next_link)
        response.raise_for_status()
        return response.json().get("value", [])
    except Exception as e:
        print(f"Error fetching annotations: {e}")
        return []

def get_job_applications(session):
    """Fetch all job applications from 2025 with their attachments using date windowing"""
    url = urljoin(CRM_BASE_URL, API_ENDPOINT)
    all_applications = []
    total_fetched = 0
    page = 1
    
    # Define date windows (one month at a time)
    date_windows = [
        ("2025-01-01T00:00:00Z", "2025-01-31T23:59:59Z"),
        ("2025-02-01T00:00:00Z", "2025-02-28T23:59:59Z"),
        ("2025-03-01T00:00:00Z", "2025-03-31T23:59:59Z"),
        ("2025-04-01T00:00:00Z", "2025-04-30T23:59:59Z"),
        ("2025-05-01T00:00:00Z", "2025-05-31T23:59:59Z"),
        ("2025-06-01T00:00:00Z", "2025-06-30T23:59:59Z"),
        ("2025-07-01T00:00:00Z", "2025-07-31T23:59:59Z"),
        ("2025-08-01T00:00:00Z", "2025-08-31T23:59:59Z"),
        ("2025-09-01T00:00:00Z", "2025-09-30T23:59:59Z"),
        ("2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z"),
        ("2025-11-01T00:00:00Z", "2025-11-30T23:59:59Z"),
        ("2025-12-01T00:00:00Z", "2025-12-31T23:59:59Z"),
    ]
    
    try:
        for start_date, end_date in date_windows:
            print(f"\nFetching applications for period: {start_date} to {end_date}")
            
            # Filter for applications created in the current date window
            filter_condition = f"createdon ge {start_date} and createdon le {end_date}"
            
            params = {
                "$select": "new_jobapplicationid,createdon",
                "$top": 5000,  # Maximum allowed
                "$orderby": "createdon desc",
                "$filter": filter_condition
            }
            
            response = session.get(url, params=params)
            response.raise_for_status()
            applications = response.json().get("value", [])
            
            if applications:
                print(f"Found {len(applications)} applications in this period")
                
                # Fetch annotations for each application
                for app in applications:
                    app_id = app["new_jobapplicationid"]
                    annotations_url = f"{url}({app_id})/new_jobapplication_Annotations?$select=filename,documentbody,mimetype"
                    print(f"Fetching annotations for application {app_id}...")
                    app["new_jobapplication_Annotations"] = get_annotations(session, annotations_url)
                
                all_applications.extend(applications)
                total_fetched += len(applications)
                print(f"Total applications fetched so far: {total_fetched}")
            else:
                print("No applications found in this period")
            
    except Exception as e:
        print(f"Error fetching job applications: {e}")
        if hasattr(e, 'response'):
            print(f"Response content: {e.response.content}")
    
    return all_applications

def download_attachments(session, applications):
    """Download CV attachments from job applications"""
    downloaded_count = 0
    skipped_count = 0
    
    for app in applications:
        try:
            app_id = app.get("new_jobapplicationid")
            created_date = app.get("createdon", "").split("T")[0]  # Get just the date part
            annotations = app.get("new_jobapplication_Annotations", [])
            
            for annotation in annotations:
                if "documentbody" in annotation and "filename" in annotation:
                    filename = annotation["filename"]
                    if filename.lower().endswith((".pdf", ".doc", ".docx")):
                        try:
                            # Generate unique filename with creation date
                            unique_filename = f"{created_date}_{filename}"
                            file_path = os.path.join(CVS_DIRECTORY, unique_filename)
                            
                            # Skip if file already exists
                            if os.path.exists(file_path):
                                print(f"Skipping existing file: {unique_filename}")
                                skipped_count += 1
                                continue
                            
                            # Decode base64 content and save
                            file_content = base64.b64decode(annotation["documentbody"])
                            with open(file_path, "wb") as f:
                                f.write(file_content)
                            print(f"Downloaded: {unique_filename}")
                            downloaded_count += 1
                        except Exception as e:
                            print(f"Error saving file {filename}: {e}")
        except Exception as e:
            print(f"Error processing application {app.get('new_jobapplicationid')}: {e}")
    
    return downloaded_count, skipped_count

def main():
    print("Starting CV download from CRM...")
    session = get_ntlm_session()
    
    # Fetch all job applications from 2025
    print("Fetching job applications from 2025...")
    applications = get_job_applications(session)
    
    if not applications:
        print("No job applications found.")
        return
    
    print(f"Found {len(applications)} job applications from 2025.")
    
    # Download CV attachments
    print("Downloading CV attachments...")
    downloaded, skipped = download_attachments(session, applications)
    
    print(f"Download process completed:")
    print(f"- Downloaded: {downloaded} new CV(s)")
    print(f"- Skipped: {skipped} existing CV(s)")
    print(f"- Total applications processed: {len(applications)}")

if __name__ == "__main__":
    # Disable SSL verification warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main() 