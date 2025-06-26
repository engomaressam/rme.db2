import mysql.connector as mysql
import pandas as pd
import traceback

LOG_FILE = "error_log.txt"

try:
    # Database connection
    cnx = mysql.connect(
        host="10.10.11.242",
        user="omaressam",
        password="Omar@1234$",
        database="RME_TEST"
    )


    if cnx.is_connected():
        print("Connected successfully!")

        # Fetch data
        query = "SELECT * FROM RME_Receipts_3"
        df = pd.read_sql(query, cnx)

        # Save to Excel
        file_name = "RME_Receipts_3.xlsx"
        df.to_excel(file_name, index=False)
        print(f"Data saved to {file_name}")

except Exception as e:
    with open(LOG_FILE, "w") as f:
        f.write("An error occurred:\n")
        f.write(traceback.format_exc())
    print(f"An error occurred. Check {LOG_FILE} for details.")

finally:
    if 'cnx' in locals() and cnx.is_connected():
        cnx.close()
