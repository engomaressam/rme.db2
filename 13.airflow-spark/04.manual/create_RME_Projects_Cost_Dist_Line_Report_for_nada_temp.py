import mysql.connector as mysql
from mysql.connector import Error

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

backup_table = "RME_Projects_Cost_Dist_Line_Report_backup_20250602"
target_table = "RME_Projects_Cost_Dist_Line_Report_for_nada_temp"

def create_table_like_backup():
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        # Drop target table if exists (optional, comment out if not desired)
        # cursor.execute(f"DROP TABLE IF EXISTS {target_table}")
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {target_table} LIKE {backup_table}")
        conn.commit()
        print(f"✅ Table {target_table} created as a copy of {backup_table}.")
        cursor.close()
        conn.close()
    except Error as e:
        print(f"❌ MySQL error: {e}")

if __name__ == "__main__":
    create_table_like_backup()
