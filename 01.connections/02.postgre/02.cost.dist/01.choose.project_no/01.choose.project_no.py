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
    WHERE project_no = '152';
""")
records = cursor.fetchall()

df = pd.DataFrame(records, columns=["project_name", "comment", "supplier_name", "amount"])
df.to_excel("cost_dist_152.xlsx", index=False)

connection.close()