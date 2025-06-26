import mysql.connector as mysql
from datetime import datetime

db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
orig_table = "RME_Projects_Cost_Dist_Line_Report"
date_str = datetime.today().strftime("%Y%m%d")
backup_table = f"{orig_table}_backup_{date_str}"

def backup_table_mysql():
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        # Drop backup table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {backup_table}")
        # Create backup table as copy
        cursor.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {orig_table}")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Backup created: {backup_table}")
    except Exception as e:
        print(f"❌ MySQL backup error: {e}")

if __name__ == "__main__":
    backup_table_mysql()
