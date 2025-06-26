# %%
import os
import psycopg2
import pandas as pd
from google.cloud import sql

# %%
os.environ["rmedb-395804"] = "your-project-id"
project_id = os.environ["rmedb-395804"]
instance_name = "rme"
database = "rme"
username = "postgres"
password = "omar_321"

connection = psycopg2.connect(
    host="34.67.241.108",
    port=5432,
    database=database,
    user=username,
    password=password,
)
cursor = connection.cursor()

# %%
cursor.execute("""
    SELECT *
    FROM cost_dist
    WHERE project_no = '152';
""")
records = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]

df = pd.DataFrame(records, columns=column_names)
df.to_excel("cost_dist_all.xlsx", index=False)

connection.close()


