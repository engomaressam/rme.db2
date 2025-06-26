import os
import oracledb
import requests

# Enable thick mode (uses Oracle Instant Client)
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")

# Database connection details
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# Microsoft Teams webhook URL
teams_webhook_url = "https://rowadmodern.webhook.office.com/webhookb2/84557fde-c0e1-457d-99d7-152442e5b0ac@7c9607e1-cd01-4c4f-a163-c7f2bb6284a4/IncomingWebhook/344902cf52cd4da2ba20168f0f1e9f7a/24f28753-9c07-40e0-91b2-ea196c200a33/V2vgTh-ifdHFN3gcme6xDjNMuHwC6MjwpAipx7SMq80Rg1"

# File paths for storing last cash-out and last message
base_dir = r"D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\13.airflow-spark\08.cash-out-message"
os.makedirs(base_dir, exist_ok=True)
last_cash_out_file = os.path.join(base_dir, "last_cash_out.txt")
last_message_file = os.path.join(base_dir, "last_message_cash_out.txt")

# Helper functions to read/write from/to text files
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

try:
    print("Connecting to the database...")
    # Connect to the database
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connected to the database!")

    # Create a cursor
    cursor = connection.cursor()

    # Query to get the latest cash-out transaction excluding unknown projects
    query = """
    SELECT 
        pap.name AS project_name,
        ASA.VENDOR_NAME AS supplier_name,
        TO_CHAR(aca.CHECK_DATE, 'YYYY-MM-DD') AS check_date,
        aca.amount AS amount
    FROM 
        APPS.AP_CHECKS_ALL ACA
        JOIN apps.ap_invoice_payments_all aip ON aip.check_id = aca.check_id
        JOIN apps.ap_invoices_all ai ON ai.invoice_id = aip.invoice_id
        JOIN apps.AP_SUPPLIERS ASA ON aca.VENDOR_ID = ASA.VENDOR_ID
        LEFT JOIN apps.PA_PROJECTS_ALL pap ON ai.ATTRIBUTE6 = pap.project_id
    WHERE 
        TO_CHAR(aca.CHECK_DATE, 'YYYY-MM-DD') BETWEEN TO_CHAR(TRUNC(SYSDATE, 'YEAR'), 'YYYY-MM-DD') 
        AND TO_CHAR(TRUNC(SYSDATE) - 1, 'YYYY-MM-DD')
        AND pap.name IS NOT NULL
        AND pap.name != 'Unknown Project'
    ORDER BY aca.CHECK_ID DESC
    FETCH FIRST 1 ROW ONLY
    """
    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        # Extract the current transaction details
        project_name = result[0]
        supplier_name = result[1]
        check_date = result[2]
        amount = float(result[3])
        current_transaction = f"{project_name}|{supplier_name}|{check_date}|{amount:.2f}"

        print(f"Fetched transaction from database: {current_transaction}")

        # Read the last cash-out and last message from files
        last_transaction = read_file(last_cash_out_file)
        last_message = read_file(last_message_file)

        # Check if the transaction is new
        if current_transaction != last_transaction:
            print("New transaction found! Preparing to send to Teams...")
            # Construct the message
            message = (
                f"New Cash-Out Transaction:\n"
                f"🏗 Project: {project_name}\n"
                f"📦 Supplier: {supplier_name}\n"
                f"📅 Check Date: {check_date}\n"
                f"💰 Amount: {amount:,.2f} EGP"
            )

            # Check if the message is already sent
            if message != last_message:
                # Send the message to Microsoft Teams
                payload = {"text": message}
                response = requests.post(teams_webhook_url, json=payload, verify=False)

                if response.status_code == 200:
                    print("Message sent to Microsoft Teams successfully!")

                    # Update the files
                    write_file(last_cash_out_file, current_transaction)
                    write_file(last_message_file, message)
                else:
                    print(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
            else:
                print("Message already sent. No duplicate message sent to Teams.")
        else:
            print("No new cash-out transactions found.")
    else:
        print("No cash-out transactions found in the database.")

except oracledb.Error as error:
    print(f"Error connecting to database: {error}")

finally:
    # Close the cursor and connection
    try:
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
    except oracledb.Error as error:
        print(f"Error while closing cursor: {error}")

    try:
        if 'connection' in locals() and connection is not None:
            connection.close()
    except oracledb.Error as error:
        print(f"Error while closing connection: {error}")
