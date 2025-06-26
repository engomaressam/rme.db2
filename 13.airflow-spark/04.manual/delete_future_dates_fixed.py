import mysql.connector as mysql
from datetime import date

db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
table = "RME_Projects_Cost_Dist_Line_Report_fixed"
today = date.today().strftime("%Y-%m-%d")

try:
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table} WHERE GL_DATE > %s", (today,))
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Deleted {deleted} rows with GL_DATE after {today} from {table}.")
except Exception as e:
    print(f"❌ MySQL error: {e}")
