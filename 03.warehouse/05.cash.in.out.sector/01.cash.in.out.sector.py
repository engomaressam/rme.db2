import mysql.connector as mysql
from mysql.connector import Error
import pandas as pd
import os

def get_table_columns(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()
    cursor.close()
    return columns

def get_unique_projects(connection, table, project_num_col, project_name_col, sector_col):
    query = f"""
        SELECT DISTINCT {project_num_col} AS PROJECT_NUM, {project_name_col} AS PROJECT_NAME, {sector_col} AS SECTOR
        FROM {table}
        WHERE {project_num_col} IS NOT NULL AND {project_name_col} IS NOT NULL AND {sector_col} IS NOT NULL
    """
    df = pd.read_sql(query, connection)
    return df

def get_table_df(connection, table, columns):
    query = f"SELECT {', '.join(columns)} FROM {table}"
    df = pd.read_sql(query, connection)
    return df

try:
    # Establish the connection
    cnx = mysql.connect(
        host="10.10.11.242",
        user="omar2",
        password="Omar_54321",
        database="RME_TEST"
    )

    if cnx.is_connected():
        print("Connection successful!\n")
        table = "SWD_Collection_Report"
        print(f"Columns for {table}:")
        columns = get_table_columns(cnx, table)
        for col in columns:
            print(col)
        print("\n" + "-"*40 + "\n")

        # Get unique projects from both tables
        df1 = get_unique_projects(cnx, "RME_Projects_Cost_Dist_Line_Report", "PROJECT_NUM", "PROJECT_NAME", "SECTOR")
        df2 = get_unique_projects(cnx, "RME_ap_check_payments_Report", "PROJECT_NUMBER", "PROJECT_NAME", "SECTOR")
        projects = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["PROJECT_NUM", "PROJECT_NAME", "SECTOR"])
        out_path = os.path.join(os.path.dirname(__file__), "unique_projects_with_sectors.csv")
        projects.to_csv(out_path, index=False)
        print(f"Saved unique projects with sectors to {out_path}")

        # Read all relevant data
        cost_df = get_table_df(cnx, "RME_Projects_Cost_Dist_Line_Report", ["PROJECT_NUM", "AMOUNT"])
        cashout_df = get_table_df(cnx, "RME_ap_check_payments_Report", ["PROJECT_NUMBER", "EQUIV"])
        cashin_df = get_table_df(cnx, "SWD_Collection_Report", ["PROJECT_NUM", "FUNC_AMOUNT"])

        # Standardize column names for merging
        cost_df = cost_df.rename(columns={"PROJECT_NUM": "PROJECT_NUM", "AMOUNT": "COST"})
        cashout_df = cashout_df.rename(columns={"PROJECT_NUMBER": "PROJECT_NUM", "EQUIV": "CASH_OUT"})
        cashin_df = cashin_df.rename(columns={"PROJECT_NUM": "PROJECT_NUM", "FUNC_AMOUNT": "CASH_IN"})

        # Aggregate sums per project
        cost_sum = cost_df.groupby("PROJECT_NUM")["COST"].sum().reset_index()
        cashout_sum = cashout_df.groupby("PROJECT_NUM")["CASH_OUT"].sum().reset_index()
        cashin_sum = cashin_df.groupby("PROJECT_NUM")["CASH_IN"].sum().reset_index()

        # Merge with project/sector info
        project_summary = projects.merge(cost_sum, on="PROJECT_NUM", how="left") \
                                 .merge(cashout_sum, on="PROJECT_NUM", how="left") \
                                 .merge(cashin_sum, on="PROJECT_NUM", how="left")
        project_summary = project_summary.fillna(0)
        project_summary = project_summary.sort_values(by=["PROJECT_NUM", "PROJECT_NAME", "SECTOR"])

        # Save per project
        out_project = os.path.join(os.path.dirname(__file__), "summary_per_project.xlsx")
        project_summary.to_excel(out_project, index=False)
        print(f"Saved per-project summary to {out_project}")

        # Aggregate per sector
        sector_summary = project_summary.groupby("SECTOR")[["COST", "CASH_OUT", "CASH_IN"]].sum().reset_index()
        out_sector = os.path.join(os.path.dirname(__file__), "summary_per_sector.xlsx")
        sector_summary.to_excel(out_sector, index=False)
        print(f"Saved per-sector summary to {out_sector}")

except Error as e:
    print(f"Error connecting to database: {e}")

finally:
    # Ensure connection is closed properly
    if 'cnx' in locals() and cnx.is_connected():
        cnx.close()
        print("Connection closed.")
