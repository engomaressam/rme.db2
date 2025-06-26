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

try:
    # Connect to the database
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connected to the database!")

    # Create a cursor
    cursor = connection.cursor()

    # Updated query to fetch the latest 4 transactions based on CHECK_ID
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
    ORDER BY aca.CHECK_ID DESC
    FETCH FIRST 4 ROWS ONLY
    """

    # Execute the query
    cursor.execute(query)
    results = cursor.fetchall()

    # Process the results to create sentences
    sentences = []
    for row in results:
        project_name = row[0] if row[0] else "Unknown Project"
        supplier_name = row[1] if row[1] else "Unknown Supplier"
        check_date = row[2]
        amount = f"{row[3]:,.2f}"  # Format amount with commas and 2 decimal places
        sentences.append(
            f"Project: {project_name}, Supplier: {supplier_name}, Check Date: {check_date}, Amount: {amount} EGP"
        )

    # Combine sentences into a single message with an empty line between transactions
    message = "Recent Cash-Out Transactions:\n\n" + "\n\n".join(sentences)
    print(message)

    # Send the message to Microsoft Teams (SSL verification bypassed here)
    payload = {"text": message}
    response = requests.post(teams_webhook_url, json=payload, verify=False)

    # Check if the message was sent successfully
    if response.status_code == 200:
        print("Message sent to Microsoft Teams successfully!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")

except oracledb.Error as error:
    print(f"Error connecting to database: {error}")

finally:
    # Ensure resources are cleaned up
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'connection' in locals() and connection:
        connection.close()
