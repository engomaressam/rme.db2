import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
target_table = "RME_Projects_Cost_Dist_Line_Report_for_nada_temp"

excel_path = r"D:\ERP Downloads\cost_2025.xlsx"

def get_mysql_columns():
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        cursor.execute(f"SHOW COLUMNS FROM {target_table}")
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return columns
    except Error as e:
        print(f"❌ MySQL error: {e}")
        return []

def get_excel_columns():
    try:
        df = pd.read_excel(excel_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        print(f"❌ Excel error: {e}")
        return []

def main():
    mysql_cols = get_mysql_columns()
    excel_cols = get_excel_columns()
    print("\nMySQL Columns:")
    print(mysql_cols)
    print("\nExcel Columns:")
    print(excel_cols)
    # Compare
    missing_in_excel = [col for col in mysql_cols if col not in excel_cols]
    extra_in_excel = [col for col in excel_cols if col not in mysql_cols]
    print("\nColumns in MySQL but missing in Excel:")
    print(missing_in_excel)
    print("\nColumns in Excel but missing in MySQL:")
    print(extra_in_excel)
    if not missing_in_excel and not extra_in_excel:
        print("\n✅ Columns match and are ready for append!")
    else:
        print("\n❌ Columns do not match. Please align columns before appending.")

if __name__ == "__main__":
    main()
