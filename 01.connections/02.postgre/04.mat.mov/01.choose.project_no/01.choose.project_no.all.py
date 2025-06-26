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
    SELECT *
    FROM mat_mov
    WHERE project_no = '152';
""")
records = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]

df = pd.DataFrame(records, columns=column_names)
df.to_excel("mat_mov_all.xlsx", index=False)

connection.close()