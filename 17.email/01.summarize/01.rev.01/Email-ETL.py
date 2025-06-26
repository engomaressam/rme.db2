import requests
import msal
from datetime import datetime, timedelta
import pytz
import urllib3
import json
import mysql.connector as mysql
from bs4 import BeautifulSoup

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === MS Graph Auth ===
client_id = "568e0691-e855-41d1-ab2b-e449aac63a2d"
authority = "https://login.microsoftonline.com/7c9607e1-cd01-4c4f-a163-c7f2bb6284a4"
scopes = ["Mail.Read"]

# Get Access Token
app = msal.PublicClientApplication(client_id=client_id, authority=authority)
flow = app.initiate_device_flow(scopes=scopes)
if "user_code" not in flow:
    print("❌ Device flow failed.")
    exit()
print(f"\n🔐 Go to {flow['verification_uri']} and enter the code: {flow['user_code']}")
result = app.acquire_token_by_device_flow(flow)

if "access_token" not in result:
    print("❌ Token acquisition failed.")
    exit()

access_token = result["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# === Email Fetch Function ===
def fetch_emails():
    today = datetime.now(pytz.UTC)
    two_months_ago = today - timedelta(days=60)
    since = two_months_ago.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = f"https://graph.microsoft.com/v1.0/me/messages?$filter=receivedDateTime ge {since}&$orderby=receivedDateTime desc&$top=50"
    emails = []

    while url:
        print(f"📥 Fetching: {url}")
        res = requests.get(url, headers=headers, verify=False)
        data = res.json()

        if "value" not in data:
            print("❌ Error fetching data:", data)
            break

        for msg in data["value"]:
            subject = msg.get("subject")
            sender = msg.get("from", {}).get("emailAddress", {}).get("address")
            received = msg.get("receivedDateTime")
            html_body = msg.get("body", {}).get("content", "")

            # Convert HTML to plain text
            soup = BeautifulSoup(html_body, "html.parser")
            plain_text = soup.get_text(separator="\n").strip()

            emails.append({
                "subject": subject or "",
                "from": sender or "",
                "received": received,
                "body": plain_text
            })

        url = data.get("@odata.nextLink")

    return emails

# === Save to text for debugging ===
def save_debug_bodies(emails):
    with open("extracted_email_bodies.txt", "w", encoding="utf-8") as f:
        for i, email in enumerate(emails, 1):
            f.write(f"\n--- Email {i} ---\n")
            f.write(f"Subject: {email['subject']}\n")
            f.write(f"From: {email['from']}\n")
            f.write(f"Date: {email['received']}\n")
            f.write(f"Body:\n{email['body']}\n")
            f.write("-" * 60 + "\n")

# === Log too-long bodies for review ===
def log_too_long_bodies(emails):
    with open("emails_with_too_long_body.txt", "w", encoding="utf-8") as f:
        for i, email in enumerate(emails, 1):
            if len(email['body']) > 65000:  # MAX length for a MySQL TEXT field
                f.write(f"\n--- Email {i} ---\n")
                f.write(f"Subject: {email['subject']}\n")
                f.write(f"From: {email['from']}\n")
                f.write(f"Date: {email['received']}\n")
                f.write(f"Body:\n{email['body']}\n")
                f.write("-" * 60 + "\n")

# === Insert into MySQL ===# 🛢️ Insert into MySQL
def insert_emails(emails):
    print("🔌 Connecting to MySQL...")
    try:
        cnx = mysql.connect(
            host="10.10.11.242",
            user="omar2",
            password="Omar_54321",
            database="RME_TEST"
        )
        cursor = cnx.cursor()
        print("✅ Connected to MySQL.")

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nada_Emails2 (
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject TEXT,
                body LONGTEXT,
                sender VARCHAR(255),
                received_date DATETIME,
                summarize TEXT
            )
        """)
        print("✅ Table ready.")

        inserted = 0
        skipped = 0
        for e in emails:
            try:
                # Check body length before insertion
                body_length = len(e["body"])
                print(f"Email subject: {e['subject']} - Body length: {body_length} characters.")

                # If the body length exceeds a certain limit, skip it
                if body_length > 65535:  # If the body is longer than 64KB (TEXT max size)
                    print(f"⚠️ Skipped Email ID {e['subject']} - Body too long.")
                    skipped += 1
                    continue

                received_dt = datetime.fromisoformat(e["received"].replace("Z", "+00:00"))
                cursor.execute("""
                    INSERT INTO nada_Emails2 (subject, body, sender, received_date, summarize)
                    VALUES (%s, %s, %s, %s, %s)
                """, (e["subject"], e["body"], e["from"], received_dt, ""))
                inserted += 1
            except Exception as ex:
                print(f"⚠️ Skipped due to error: {ex}")

        cnx.commit()
        print(f"✅ Inserted {inserted} emails. Skipped {skipped} due to size.")

    except mysql.Error as e:
        print(f"❌ Database Error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()
            print("🔒 SQL connection closed.")


# === Main Run ===
emails = fetch_emails()
save_debug_bodies(emails)  # Save to a file for debugging
log_too_long_bodies(emails)  # Log too long bodies
insert_emails(emails)  # Insert into DB
