import pymysql

HOST = '10.10.11.242'
USER = 'omar2'
PASSWORD = 'Omar_54321'
DB = 'RME_TEST'
TABLE = 'po.pdfs'

conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DB, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()

print('COLUMNS:')
cur.execute(f'SHOW COLUMNS FROM `{TABLE}`')
for row in cur.fetchall():
    print(row)

print('\nDOCUMENT TYPES:')
cur.execute(f'SELECT DISTINCT `document_type` FROM `{TABLE}`')
for row in cur.fetchall():
    print(row)

cur.close()
conn.close()
