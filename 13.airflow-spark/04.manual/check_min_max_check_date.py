import mysql.connector as mysql
from datetime import datetime

db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
table = "RME_ap_check_payments_Report"

def main():
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT MIN(CHECK_DATE), MAX(CHECK_DATE), COUNT(*) FROM {table}")
    min_date, max_date, count = cursor.fetchone()
    print(f"MIN(CHECK_DATE): {min_date}")
    print(f"MAX(CHECK_DATE): {max_date}")
    print(f"Total rows: {count}")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
