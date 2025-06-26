
import psycopg2
import pandas as pd

connection = psycopg2.connect(
    host="localhost",
    dbname="rme",
    user="postgres",
    password="omar_321"
)
cursor = connection.cursor()

cursor.execute("""
    SELECT project_name, comment, supplier_name, amount
    FROM cost_dist
    WHERE project_name = 'EMAAR-PKG#53-UPTOWN';
""")
records = cursor.fetchall()

df = pd.DataFrame(records, columns=["project_name", "comment", "supplier_name", "amount"])
df.to_excel("cost_dist_EMAAR-PKG#53-UPTOWN.xlsx", index=False)

connection.close()