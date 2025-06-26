import psycopg2
import pandas as pd

connection = psycopg2.connect(
    host="localhost",
    dbname="rme",
    user="postgres",
    password="omar_321"
)
cursor= connection.cursor()

cursor.execute("""
    SELECT *
    FROM cost_dist
    WHERE project_no IN ('122', '130', '143', '152', '157', '166', '167', '168', '169', '189', '190')
    ORDER BY project_no ASC;
""")
records = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]

df = pd.DataFrame(records, columns=column_names)
df.to_excel("cost_dist_bridges_all.xlsx", index=False)

cursor.close()
connection.close()

