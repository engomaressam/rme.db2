import mysql.connector as mysql
import pandas as pd

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
table = "RME_Projects_Cost_Dist_Line_Report"

try:
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    # Get the most recent, oldest, and 10 largest GL_DATEs (as string)
    cursor.execute(f"SELECT GL_DATE FROM {table} WHERE GL_DATE IS NOT NULL")
    dates = [row[0] for row in cursor.fetchall()]
    df = pd.DataFrame(dates, columns=["GL_DATE"])
    df["GL_DATE"] = pd.to_datetime(df["GL_DATE"], errors='coerce')
    print("Top 10 latest GL_DATEs:")
    print(df.sort_values("GL_DATE", ascending=False).head(10))
    print("\nTop 10 earliest GL_DATEs:")
    print(df.sort_values("GL_DATE", ascending=True).head(10))
    print("\nUnique years:", sorted(df["GL_DATE"].dt.year.dropna().unique()))
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ MySQL error: {e}")
