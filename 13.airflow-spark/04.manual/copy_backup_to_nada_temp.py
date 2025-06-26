import mysql.connector as mysql
from mysql.connector import Error

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

backup_table = "RME_Projects_Cost_Dist_Line_Report_backup_20250602"
target_table = "RME_Projects_Cost_Dist_Line_Report_for_nada_temp"

def copy_backup_to_temp():
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO {target_table} SELECT * FROM {backup_table}")
        conn.commit()
        print(f"✅ All rows copied from {backup_table} to {target_table}.")
        cursor.close()
        conn.close()
    except Error as e:
        print(f"❌ MySQL error: {e}")

if __name__ == "__main__":
    copy_backup_to_temp()
