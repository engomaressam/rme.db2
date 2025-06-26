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
teams_webhook_url = "https://rowadmodern.webhook.office.com/webhookb2/84557fde-c0e1-457d-99d7-152442e5b0ac@7c9607e1-cd01-4c4f-a163-c7f2bb6284a4/IncomingWebhook/2ea28de8d34b417b8ded9774788916fd/24f28753-9c07-40e0-91b2-ea196c200a33/V2f9usIcwbZeFI--I8y4MnuI-SVjXJVlrVqccubKwdsIg1"

# File paths for storing last cash-in and last message
base_dir = r"D:\OneDrive\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\13.airflow-spark\07.cash-in-message"
os.makedirs(base_dir, exist_ok=True)
last_cash_in_file = os.path.join(base_dir, "last_cash_in.txt")
last_message_file = os.path.join(base_dir, "last_message.txt")

# Project mapping dictionary (preloaded from the Excel file)
project_mapping = {
    "0019": "Ismalia Bridge",
    "0030": "Mall Of Egypt",
    "0032": "Sodic Club House",
    "0040": "Al Mostathmreen GIS Substation",
    "0046": "El Amal Bridge",
    "0048": "Wall - New Giza",
    "0051": "Tamey El-amdeed Substation",
    "0052": "Ismailiya East Substation",
    "0053": "Amal Bridge Lock & Load",
    "0059": "New Capital Tunnels",
    "0060": "Ministries Buildings",
    "0061": "Beni Suef Substation R61",
    "0063": "Royal City",
    "0064": "Shubra-Banha lock&load",
    "0065": "El Sewedy Stores Project",
    "0066": "El Mostaqbal lock & load",
    "0068": "Kayan wall lock & Load",
    "0069": "Suez Steel Factory",
    "0070": "Hyper El-Temsah",
    "0071": "EMAAR-PKG117- MARASSI",
    "0072": "Siemens Power Station",
    "0073": "EMAAR-PKG#85-UPTOWN",
    "0074": "Tunnel of Sokhna Road",
    "0075": "Abu Sultan Road Extension",
    "0076": "Mintra New Factory",
    "0077": "King Farouk Resthouse Rest.",
    "0078": "Mohamed Ali Palace Restoration",
    "0079": "New bridge Ismailiya Nefisha",
    "0080": "Fish Market",
    "0081": "AL-JAZI EGYPT",
    "0082": "R5 Mix-Use Complex Project",
    "0083": "TSK Solar Benban",
    "0084": "Alamein",
    "0085": "Elsewedy Univ - Enabling Works",
    "0086": "Lock&Load-30June w Sokhna Rd",
    "0087": "NUCA R05 - Z03",
    "0088": "NUCA R05 - Z02",
    "0089": "AbuSultan Rd Bridge2 Extension",
    "0090": "Tunnel of Sokhna Road (2)",
    "0091": "Middle Tunnel Lock Load",
    "0092": "EDNC Retail & Offices Civil",
    "0093": "EMAAR-PKG# 144, Marassi",
    "0094": "DPW Onshore Port & Terminal",
    "0095": "EMAAR-PKG# 101-UPTOWN",
    "0096": "Substation Elco Steel",
    "0097": "Agric Greenhouses Tunnel",
    "0098": "EMAAR- Pkg 140-ITP-Mivida",
    "0099": "Olympic Multi – Sports Hall",
    "0100": "Emergency Bridge",
    "0101": "Pyramid tunnel",
    "0102": "ELCO STEEL-EGAT",
    "0103": "Abu Ghazala Lock & Load",
    "0104": "Mohamed Aly Fahmy Lock&Load",
    "0105": "El Moshier ( Abu Zaid Khedr)",
    "0106": "ORA Zed Park LOCK&LOAD",
    "0107": "DoubleTree Mangroovy ElGouna",
    "0108": "Baron Fence",
    "0109": "Safeer Square Bridge",
    "0110": "Suez Road Tunnel RES",
    "0111": "HyperOne Zayed Extension",
    "0112": "IKEA Extension MoA",
    "0113": "CFC Podium 2",
    "0114": "Mostafa Kamel Bridge L&L",
    "0115": "Joseph Tito Bridge L&L",
    "0116": "ESPIKO Bridge L&L",
    "0117": "ElGabal AlAsfar Tunnel L&L",
    "0118": "MR3 Bridge L&L",
    "0119": "ESU Ph2-Enabling & Struc",
    "0120": "Mehwar elsalam Lock & Load",
    "0121": "EL-Hegaz Square Bridge",
    "0122": "RING ROAD MARYOTIA EXPANSION",
    "0123": "EMAAR-Pkg#162/163- Marassi",
    "0124": "Sultana Malak Restoration",
    "0125": "Lekela 250MW Wind Farm",
    "0126": "New Giza Teaching Hospital",
    "0127": "Kayan Landscape",
    "0128": "BridgeNasr Rd w Abbas El-Akkad",
    "0129": "Asher Mn Ramadan Bridge No2",
    "0130": "Kafr Shokr Bridge",
    "0131": "Shorouk Bridge-LRT",
    "0132": "RING ROAD BRIDGE - El MARG",
    "0133": "GOV2 - Infra",
    "0134": "Cairo-Alex Railway",
    "0135": "Hassan El Mamoun Bridge",
    "0136": "Sherouk Bridge- LOCK&LOAD",
    "0137": "ORA ZED - Ph 01B - Pkgs A&D",
    "0138": "Kattameya Creeks",
    "0139": "Faculty of Medicine",
    "0140": "Diplomatic District - Infra",
    "0141": "ElSewedy HQ Internal Finishing",
    "0142": "BKG#178-Lagoon Discharge",
    "0143": "El Khatatba Bridge",
    "0144": "EGAT Pelletizing Plant",
    "0145": "Sokhna Port Expansion",
    "0146": "Waldorf Astoria Mock-up Room",
    "0147": "MDF Factory",
    "0148": "Olympic City Lock&Load",
    "0149": "Kemet Building",
    "0150": "Air Defense College",
    "0151": "ORA ZED-Ph 2-Pkgs A&D",
    "0152": "HST Bridges-Sokhna & Mahager",
    "0153": "October Dry Port Railway",
    "0154": "Alfa New Central Labs",
    "0155": "R06 Loack & Load",
    "0156": "Waslet Om Amar Bridge",
    "0157": "Abou Ghaleb Bridge",
    "0158": "Egyptian Exchange Building",
    "0159": "Port Said Port Silos",
    "0160": "The Open Channel Project",
    "0161": "Ras El Teen Hangar",
    "0162": "Asmarat Roads L&L",
    "0163": "El Shohadaa Mosque No.006",
    "0164": "Waldorf Astoria Cairo",
    "0165": "King Mariout  Bridge",
    "0166": "Alamein Coastal Road Bridge",
    "0167": "Wady El Natroon Bridge",
    "0168": "HST El Mahager Bridge",
    "0169": "HST Culverts",
    "0170": "Mivida BP#189",
    "0171": "EDNC Hardscape Package",
    "0172": "Egat Rolling Mill no.4",
    "0173": "EGAT Lock & Load",
    "0174": "Qani bay Al rammah Mosque",
    "0175": "October Under-Railway Tunnel",
    "0176": "Radamis City",
    "0177": "SODIC Allegria Villa f100",
    "0178": "El- Hussein Mosque",
    "0179": "Jawhar Al-Lala Mosque",
    "0180": "Endowments Building",
    "0181": "Beymen Fit Out",
    "0182": "Astoria Sharm elSheikh",
    "0183": "Wadi Halfa Port",
    "0184": "SSC Suez Steel Company Project"
}

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

    # Query to get the latest cash-in transaction
    query_latest_transaction = """
    SELECT acr.CASH_RECEIPT_ID,
           TO_CHAR(acr.RECEIPT_DATE, 'YYYY-MM-DD') AS receipt_date,
           TRUNC(acr.amount * NVL(acr.exchange_rate, 1)) AS Recipt_amount_EGP,
           acr.comments,
           acr.ATTRIBUTE1  -- Project number
    FROM AR_CASH_RECEIPTS_ALL acr
    WHERE acr.ATTRIBUTE_CATEGORY = '83'
      AND acr.ATTRIBUTE1 IS NOT NULL
      AND acr.ATTRIBUTE2 != 'Manual/Netting'
    ORDER BY acr.CASH_RECEIPT_ID DESC
    FETCH FIRST 1 ROW ONLY
    """
    
    cursor.execute(query_latest_transaction)
    result = cursor.fetchone()

    if result:
        # Extract the latest transaction details
        cash_receipt_id, receipt_date, amount_egp, comment, project_number = result
        
        # Extract project name from mapping, if available
        project_name = project_mapping.get(project_number, None)

        # Query to get the total cash-in for the current month
        query_total_month = f"""
        SELECT SUM(TRUNC(acr.amount * NVL(acr.exchange_rate, 1))) AS total_month
        FROM AR_CASH_RECEIPTS_ALL acr
        WHERE acr.ATTRIBUTE_CATEGORY = '83'
          AND acr.ATTRIBUTE1 IS NOT NULL
          AND acr.ATTRIBUTE2 != 'Manual/Netting'
          AND TO_CHAR(acr.RECEIPT_DATE, 'YYYY-MM') = TO_CHAR(SYSDATE, 'YYYY-MM')
        """
        
        cursor.execute(query_total_month)
        total_month_result = cursor.fetchone()
        total_month = total_month_result[0] if total_month_result[0] else 0

        # Construct transaction identifier
        current_transaction = f"{cash_receipt_id}|{receipt_date}|{amount_egp}|{comment.strip()}"
        print(f"Fetched transaction from database: {current_transaction}")

        # Read the last cash-in stored locally
        last_transaction = read_file(last_cash_in_file)

        if current_transaction != last_transaction:
            print("New transaction found! Preparing to send to Teams...")
            
            # Construct the message
            message = "New Cash-In Transaction:\n"
            if project_name:  # Only include project if it's found
                message += f"🏗️ Project: {project_name}\n"
            message += (
                f"📅 Date: {receipt_date}\n"
                f"💰 Amount: {amount_egp:,} EGP\n"
                f"📝 Comment: {comment if comment else 'No comment provided'}\n"
                f"🇪🇬 Total Cash-In for {receipt_date[:7]}: {total_month:,.2f} EGP"
            )

            # Check if the message is already sent
            last_message = read_file(last_message_file)
            if message != last_message:
                # Send the message to Microsoft Teams
                payload = {"text": message}
                response = requests.post(teams_webhook_url, json=payload, verify=False)

                if response.status_code == 200:
                    print("Message sent to Microsoft Teams successfully!")

                    # Update the files
                    write_file(last_cash_in_file, current_transaction)
                    write_file(last_message_file, message)
                else:
                    print(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
            else:
                print("Message already sent. No duplicate message sent to Teams.")
        else:
            print("No new cash-in transactions found.")
    else:
        print("No cash-in transactions found in the database.")

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
