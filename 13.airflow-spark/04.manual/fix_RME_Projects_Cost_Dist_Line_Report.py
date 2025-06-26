import mysql.connector as mysql
from datetime import datetime

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
orig_table = "RME_Projects_Cost_Dist_Line_Report"
fixed_table = orig_table + "_fixed"

def get_columns_and_types():
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"DESCRIBE {orig_table}")
    columns = cursor.fetchall()
    cursor.close()
    conn.close()
    return columns

def create_fixed_table(columns):
    # Build CREATE TABLE statement, converting date-like longtext columns to DATE
    col_defs = []
    date_like_cols = []
    for col in columns:
        name, dtype = col[0], col[1].lower()
        if dtype.startswith('longtext') and 'date' in name.lower():
            col_defs.append(f"`{name}` DATE NULL")
            date_like_cols.append(name)
        else:
            col_defs.append(f"`{name}` {dtype} NULL")
    col_defs_str = ",\n    ".join(col_defs)
    create_sql = f"CREATE TABLE {fixed_table} (\n    {col_defs_str}\n)"
    return create_sql, date_like_cols

def copy_and_convert_data(columns, date_like_cols):
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    # Build column list for insert
    col_names = [col[0] for col in columns]
    select_exprs = []
    for name in col_names:
        if name in date_like_cols:
            select_exprs.append(f"STR_TO_DATE(`{name}`, '%Y-%m-%d') AS `{name}`")
        else:
            select_exprs.append(f"`{name}`")
    select_sql = f"SELECT {', '.join(select_exprs)} FROM {orig_table}"
    insert_sql = f"INSERT INTO {fixed_table} ({', '.join([f'`{c}`' for c in col_names])}) {select_sql}"
    cursor.execute(insert_sql)
    conn.commit()
    cursor.close()
    conn.close()

def main():
    print(f"🔄 Getting columns and types from {orig_table}...")
    columns = get_columns_and_types()
    print(f"🔄 Creating fixed table {fixed_table} with date conversion...")
    create_sql, date_like_cols = create_fixed_table(columns)
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {fixed_table}")
    cursor.execute(create_sql)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"🔄 Copying and converting data...")
    copy_and_convert_data(columns, date_like_cols)
    print(f"✅ Fixed table {fixed_table} created with date columns converted.")

if __name__ == "__main__":
    main()
