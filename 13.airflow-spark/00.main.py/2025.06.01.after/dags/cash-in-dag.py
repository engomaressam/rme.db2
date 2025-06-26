import os
import oracledb
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pendulum

# ✅ Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ✅ Database connection details
oracledb.init_oracle_client(lib_dir="/usr/lib/oracle/19/client64")  # Ensure the path is correct for Linux
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# ✅ Microsoft Teams Webhook
teams_webhook_url = "https://rowadmodern.webhook.office.com/webhookb2/84557fde-c0e1-457d-99d7-152442e5b0ac@7c9607e1-cd01-4c4f-a163-c7f2bb6284a4/IncomingWebhook/2ea28de8d34b417b8ded9774788916fd/24f28753-9c07-40e0-91b2-ea196c200a33/V2f9usIcwbZeFI--I8y4MnuI-SVjXJVlrVqccubKwdsIg1"

# ✅ Project mapping dictionary
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

# ✅ File paths (for Linux system)
base_dir = "/home/PMO/Downloads/cash-in-message"  # Ensure this is the correct path on your system
os.makedirs(base_dir, exist_ok=True)
last_cash_in_file = os.path.join(base_dir, "last_cash_in.txt")

# ✅ Read/Write helper functions
def read_file(file_path):
    """Reads content from a file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

def write_file(file_path, content):
    """Writes content to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

# ✅ ETL function for Oracle to Teams message
def erp_to_mysql_etl():
    try:
        print("🔄 Setting up Oracle connection...")

        # ✅ Connect to Oracle ERP
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor_erp = connection_erp.cursor()

        print("✅ Connected to Oracle ERP!")

        # ✅ Fetch the last 15 cash-in transactions (most recent first)
        query_latest_transactions = """
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
        FETCH FIRST 15 ROWS ONLY
        """
        cursor_erp.execute(query_latest_transactions)
        results = cursor_erp.fetchall()

        # ✅ Read last processed transactions (Only cash receipt IDs)
        last_transaction = read_file(last_cash_in_file)
        processed_transactions = set(last_transaction.splitlines()) if last_transaction else set()

        new_transactions = []

        for result in reversed(results):  # Ensure chronological order
            cash_receipt_id, receipt_date, amount_egp, comment, project_number = result

            if str(cash_receipt_id) not in processed_transactions:
                new_transactions.append((cash_receipt_id, receipt_date, amount_egp, comment, project_number))

        if new_transactions:
            print(f"New transactions found: {len(new_transactions)}. Preparing to send notifications...")

            # ✅ Construct a single message containing all new transactions
            message = "🆕 New Cash-In Transactions:\n\n"

            for cash_receipt_id, receipt_date, amount_egp, comment, project_number in new_transactions:
                project_name = project_mapping.get(project_number, None)  # Get project name if available

                message += (
                    f"⏳ Date: {receipt_date}\n"
                    + (f"🏗️ Project: {project_name}\n" if project_name else "")
                    + f"💰 Amount: {amount_egp:,} EGP\n"
                    + f"📝 Comment: {comment if comment else 'No comment provided'}\n\n"
                )

            # ✅ Get current date information
            today = datetime.now()
            current_month = today.strftime("%b-%y")  # e.g., "Mar-25"
            previous_month = (today.replace(day=1) - timedelta(days=1)).strftime("%b-%y")  # e.g., "Feb-25"

            # ✅ Fetch total cash-in for the current month
            query_current_month = """
            SELECT SUM(TRUNC(acr.amount * NVL(acr.exchange_rate, 1))) AS total_month
            FROM AR_CASH_RECEIPTS_ALL acr
            WHERE acr.ATTRIBUTE_CATEGORY = '83'
            AND acr.ATTRIBUTE1 IS NOT NULL
            AND acr.ATTRIBUTE2 != 'Manual/Netting'
            AND acr.REVERSAL_REASON_CODE IS NULL
            AND TO_CHAR(acr.RECEIPT_DATE, 'YYYY-MM') = TO_CHAR(SYSDATE, 'YYYY-MM')
            """
            cursor_erp.execute(query_current_month)
            current_total_result = cursor_erp.fetchone()
            total_current_month = current_total_result[0] if current_total_result[0] else 0

            # ✅ If today is between 1st and 10th, fetch previous month's total
            if today.day <= 10:
                query_previous_month = """
                SELECT SUM(TRUNC(acr.amount * NVL(acr.exchange_rate, 1))) AS total_month
                FROM AR_CASH_RECEIPTS_ALL acr
                WHERE acr.ATTRIBUTE_CATEGORY = '83'
                AND acr.ATTRIBUTE1 IS NOT NULL
                AND acr.ATTRIBUTE2 != 'Manual/Netting'
                AND acr.REVERSAL_REASON_CODE IS NULL
                AND TO_CHAR(acr.RECEIPT_DATE, 'YYYY-MM') = TO_CHAR(ADD_MONTHS(SYSDATE, -1), 'YYYY-MM')
                """
                cursor_erp.execute(query_previous_month)
                previous_total_result = cursor_erp.fetchone()
                total_previous_month = previous_total_result[0] if previous_total_result[0] else 0

                # ✅ Display both previous and current month totals
                message += "────────────────────────\n"
                message += f"🇪🇬 Total Cash-In for {previous_month}: {total_previous_month:,.2f} EGP\n"
                message += f"🇪🇬 Total Cash-In for {current_month}: {total_current_month:,.2f} EGP\n"

            else:
                # ✅ Display only the current month's total
                message += "────────────────────────\n"
                message += f"🇪🇬 Total Cash-In for {current_month}: {total_current_month:,.2f} EGP\n"

            # ✅ Save the message to a text file
            transaction_file_path = os.path.join(base_dir, "new_cash_in_transactions.txt")
            write_file(transaction_file_path, message)
            print(f"✅ Transactions saved to {transaction_file_path}")

            # ✅ Send message to Microsoft Teams (optional)
            payload = {"text": message}
            response = requests.post(teams_webhook_url, json=payload, verify=False)
            if response.status_code == 200:
                print("✅ Message sent to Teams!")
            else:
                print(f"❌ Failed to send message to Teams: {response.status_code}, {response.text}")

            # ✅ Save only the last 15 cash receipt IDs
            latest_transactions = [str(t[0]) for t in results]
            write_file(last_cash_in_file, "\n".join(latest_transactions))

        else:
            print("No new cash-in transactions found.")

    except oracledb.Error as error:
        print(f"❌ Error connecting to database: {error}")

    finally:
        if 'cursor_erp' in locals():
            cursor_erp.close()
        if 'connection_erp' in locals():
            connection_erp.close()
# ✅ Airflow DAG Configuration
local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'gamal',
    'start_date': datetime(2025, 5, 7, tzinfo=local_tz),
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
    'email': ['mohamed.Ghassan@rowad-rme.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'cash_in',
    catchup=False,
    default_args=default_args,
    schedule_interval='*/10 * * * *',  # Run every 10 minutes
    tags=['ERP', 'Oracle']
)

# ✅ Define the PythonOperator to run the cash_in ETL task
cash_in_task = PythonOperator(
    dag=dag,
    task_id='cash_in_etl_task',  # Correct task name
    python_callable=erp_to_mysql_etl  # Your ETL function
)

# ✅ Run the task
cash_in_task
