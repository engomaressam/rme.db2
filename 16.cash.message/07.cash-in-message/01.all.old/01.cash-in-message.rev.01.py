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
teams_webhook_url = "https://rowadmodern.webhook.office.com/webhookb2/84557fde-c0e1-457d-99d7-152442e5b0ac@7c9607e1-cd01-4c4f-a163-c7f2bb6284a4/IncomingWebhook/2ea28de8d34b417b8ded9774788916fd/24f28753-9c07-40e0-91b2-ea196c200a33/V2f9usIcwbZeFI--I8y4MnuI-SVjXJVlrVqccubKwdsIg1"

try:
    # Connect to the database
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connected to the database!")

    # Create a cursor
    cursor = connection.cursor()

    # Query to get the latest 2 receipt dates with filtering conditions
    query = """
    SELECT TO_CHAR(acr.RECEIPT_DATE, 'YYYY-MM-DD') AS receipt_date,
           TRUNC(acr.amount * NVL(acr.exchange_rate, 1)) AS Recipt_amount_EGP,
           acr.comments
    FROM AR_CASH_RECEIPTS_ALL acr
    WHERE acr.ATTRIBUTE_CATEGORY = '83'
      AND acr.ATTRIBUTE1 IS NOT NULL
      AND acr.ATTRIBUTE2 != 'Manual/Netting'
    ORDER BY acr.RECEIPT_DATE DESC
    FETCH FIRST 2 ROWS ONLY
    """

    # Execute the query
    cursor.execute(query)
    results = cursor.fetchall()

    # Process the results to create the sentences
    sentences = []
    for row in results:
        receipt_date = row[0]  # Already formatted as YYYY-MM-DD
        amount_egp = f"{int(row[1]):,}"  # Format the amount with commas and no decimals
        comment = row[2]
        sentences.append(
            f"On {receipt_date}, a cash-in of {amount_egp} EGP was recorded with the comment: '{comment}'."
        )

    # Combine sentences into a single message with an empty line between transactions
    message = "\n\n".join(sentences)
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
    # Close the cursor and connection
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'connection' in locals() and connection:
        connection.close()
