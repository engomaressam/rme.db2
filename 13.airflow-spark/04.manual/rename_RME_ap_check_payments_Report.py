import mysql.connector as mysql

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

try:
    print("🔄 Connecting to MySQL...")
    mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    mysql_cursor = mysql_connection.cursor()
    print("✅ Connected. Renaming table...")
    mysql_cursor.execute("RENAME TABLE RME_ap_check_payments_Report TO RME_ap_check_payments_Report_old")
    mysql_connection.commit()
    print("✅ Table renamed to RME_ap_check_payments_Report_old.")
    mysql_cursor.close()
    mysql_connection.close()
except Exception as e:
    print(f"❌ MySQL error: {e}")
