import os
import oracledb
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from datetime import datetime, timedelta

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Oracle ERP connection details
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
mysql_table = "RME_Projects_Cost_Dist_Line_Report_new"

# The core Oracle query, parameterized for a single day
def get_oracle_query(gl_date):
    return f"""
    SELECT 
	dis_ln.expenditure_item_id expenditure_item_id,
	pts.user_transaction_source transaction_source,
	dis_ln.project_id project_id,
	prj.segment1 Project_Num,
	prj.name Project_Name,
	prj.project_type project_type,
	dis_ln.task_id task_id,
	tsk.task_number Task_Num,
	tsk.task_name Task_Name,
	tsk.work_type_id bu_id,
	tsk.top_task_id,
	tsk.service_type_code bl_id,
	tl.attribute1 Project,
	tl.attribute2 FLOOR,
	tl.attribute3 Area,
	A.SECTOR,
	pol1.attribute2 AREAS,
	to_char(dis_ln.gl_date, 'YYYY-MM-DD') expenditure_item_date,
	exp_itm.expenditure_type,
	ota_general.get_org_name(NVL(exp_itm.override_to_organization_id, pa_exp.incurred_by_organization_id)) expenditure_org_name,
	dis_ln.amount,
	dis_ln.dr_code_combination_id,
	dis_ln.cr_code_combination_id,
	to_char(dis_ln.transferred_date, 'YYYY-MM-DD') transferred_date,
	dis_ln.transfer_status_code,
	dis_ln.acct_event_id,
	dis_ln.batch_name,
	dis_ln.transfer_rejection_reason,
	inv.invoice_num,
	VSIT.ADDRESS_LINE1 SUPPLIER_SITE,
	exp_itm.document_header_id,
	dis_ln.system_reference2,
	com.EXPENDITURE_COMMENT,
	sub.SEGMENT1 vendor_number,
	sub.vendor_name vendor_name,
	to_char(dis_ln.gl_date, 'YYYY-MM-DD') gl_date,
	exp_itm.ORIG_TRANSACTION_REFERENCE IPC_NO,
	NULL concat_seg_cr,
	NULL concat_seg_dr,
	NVL(po1.segment1, po2.segment1) po_number,
	sys.SEGMENT1 ITEM_CODE,
	NVL(NVL((SELECT max(SEGMENT_VALUE)
                          FROM apps.PA_SEGMENT_VALUE_LOOKUPS SS
                         WHERE SEGMENT_VALUE_LOOKUP_SET_ID = 507
                           AND SEGMENT_VALUE_LOOKUP =
                               (SELECT SUBSTR(SEGMENT1, 0, 2)
                                  FROM apps.mtl_system_items_B
                                 WHERE INVENTORY_ITEM_ID =
                                       exp_itm.INVENTORY_ITEM_ID
                                   AND ROWNUM = 1)),
                        (SELECT max(ASS_ATTRIBUTE1)
                           FROM apps.PER_ALL_ASSIGNMENTS_F
                          WHERE SYSDATE BETWEEN EFFECTIVE_START_DATE AND
                                EFFECTIVE_END_DATE
                            AND PERSON_ID = (SELECT POH.AGENT_ID
                                               FROM apps.PO_HEADERS_ALL POH
                                              WHERE POH.PO_HEADER_ID =
                                                    NVL(NVL(inv.quick_po_header_id,
                                                            inv.attribute5),
                                                        exp_itm.document_header_id)
                                                AND ROWNUM = 1))),
                    'Finacial') owner,
	NVL(inv.quick_po_header_id, inv.attribute5) x1,
	exp_itm.document_header_id x2,
	exp_itm.Quantity Quantity,
	rme_dev.pa_utils4.get_unit_of_measure_m(exp_itm.unit_of_measure, exp_itm.expenditure_type) UOM,
	NVL((SELECT  max(ss.line_no)
                      FROM apps.rme_prj_cont_lines ss,
                           apps.rme_prj_cont_wc wc
                     WHERE exp_itm.ORIG_TRANSACTION_REFERENCE = wc.ipc_no
                       AND ss.wc_id = wc.wc_id
                       AND prj.project_id = ss.project_id
                       AND ROWNUM = 1),
                    dis_ln.line_num) line_num,
	NVL((SELECT  max(ss.LINE_DESC)
                      FROM apps.rme_prj_cont_lines ss,
                           apps.rme_prj_cont_wc wc
                     WHERE exp_itm.ORIG_TRANSACTION_REFERENCE = wc.ipc_no
                       AND ss.wc_id = wc.wc_id
                       AND prj.project_id = ss.project_id
                       AND ROWNUM = 1),
                    (SELECT EXPENDITURE_COMMENT
                       FROM apps.pa_expenditure_comments
                      WHERE EXPENDITURE_ITEM_ID = exp_itm.EXPENDITURE_ITEM_ID
                        AND rownum = 1)) LINE_DESC
    FROM
	apps.pa_cost_distribution_lines_all dis_ln
    LEFT JOIN apps.pa_projects_all prj ON
	    dis_ln.project_id = prj.project_id
    LEFT JOIN apps.hr_organization_units org ON
	    prj.carrying_out_organization_id = org.organization_id
    LEFT JOIN apps.gl_code_combinations gl ON
	    dis_ln.dr_code_combination_id = gl.code_combination_id
    LEFT JOIN apps.pa_tasks tsk ON
	    dis_ln.task_id = tsk.task_id
	    AND tsk.project_id = prj.project_id
    LEFT JOIN apps.pa_expenditure_items_all exp_itm ON
	    exp_itm.expenditure_item_id = dis_ln.expenditure_item_id
    LEFT JOIN apps.gl_code_combinations cr_gl ON
	    dis_ln.cr_code_combination_id = cr_gl.code_combination_id
    LEFT JOIN apps.pa_expenditures_all pa_exp ON
	    exp_itm.expenditure_id = pa_exp.expenditure_id
    LEFT JOIN apps.ap_invoices_all inv ON
	    exp_itm.document_header_id = inv.invoice_id
    LEFT JOIN apps.pa_transaction_sources pts ON
	    exp_itm.transaction_source = pts.transaction_source
    LEFT JOIN apps.AP_SUPPLIER_SITES_ALL VSIT ON
	    VSIT.VENDOR_SITE_ID = INV.VENDOR_SITE_ID
	    AND rownum = 1
    LEFT JOIN apps.pa_expenditure_comments com ON
	    com.EXPENDITURE_ITEM_ID = exp_itm.EXPENDITURE_ITEM_ID
	    AND rownum = 1
    LEFT JOIN apps.PO_VENDORS sub ON
	    sub.vendor_id = NVL(exp_itm.VENDOR_ID, pa_exp.VENDOR_ID)
	    AND rownum = 1
    LEFT JOIN apps.po_headers_all po1 ON
	    po1.po_header_id = exp_itm.document_header_id
	    AND rownum = 1
    LEFT JOIN apps.po_headers_all po2 ON
	    po2.po_header_id = NVL(inv.quick_po_header_id, inv.attribute5)
	    AND rownum = 1
    LEFT JOIN (
	    SELECT
		    PROJECT_ID,
		    max(sector) sector
	    FROM
		    APPS.RME_PROJECT_SECTORS
	    GROUP BY
		    PROJECT_ID) A ON
	    A.PROJECT_ID = prj.PROJECT_ID
    LEFT JOIN (
	    SELECT
		    POL.org_id,
		    POL.PO_HEADER_ID,
		    max(pol.attribute2) attribute2
	    FROM
		    apps.PO_LINE_LOCATIONS_ALL pol
	    GROUP BY
		    POL.org_id,
		    POL.PO_HEADER_ID
    ) pol1 ON
	    pol1.PO_HEADER_ID = inv.attribute5
	    AND POL1.org_id = dis_ln.org_id
    LEFT JOIN apps.mtl_system_items_b sys ON
	    sys.INVENTORY_ITEM_ID = exp_itm.INVENTORY_ITEM_ID
	    AND sys.organization_id = org.organization_id
    LEFT JOIN apps.mtl_material_transactions mmt ON
	    mmt.Inventory_ITEM_ID = exp_itm.Inventory_ITEM_ID
	    AND exp_itm.orig_transaction_reference =
                            CAST(mmt.transaction_id AS VARCHAR(255))
    LEFT JOIN apps.mtl_txn_request_lines tl ON
	    tl.line_id = mmt.move_order_line_id
	    AND tl.task_id = tsk.task_id
    WHERE        
	    dis_ln.amount != 0
        and to_char(dis_ln.gl_date, 'YYYY-MM-DD') = '{gl_date}'
    """

def ensure_mysql_table_exists(df, table_name):
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cols = []
    for col in df.columns:
        # Use TEXT for all columns for maximum compatibility
        cols.append(f"`{col}` TEXT NULL")
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {', '.join(cols)}\n)"
    cursor.execute(create_sql)
    conn.commit()
    cursor.close()
    conn.close()

def insert_to_mysql(df, table_name):
    if df.empty:
        print("❌ No data to insert.")
        return
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cols = [f'`{c}`' for c in df.columns]
    insert_sql = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))})"
    cursor.executemany(insert_sql, df.astype(str).where(pd.notnull(df), None).values.tolist())
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into {table_name}.")

def fetch_oracle_data(gl_date):
    try:
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor = conn.cursor()
        query = get_oracle_query(gl_date)
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        cursor.close()
        conn.close()
        print(f"✅ Oracle: {len(df)} rows fetched for {gl_date}")
        return df
    except Exception as e:
        print(f"❌ Oracle error: {e}")
        return pd.DataFrame()

def get_last_gl_date_mysql(table_name):
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(GL_DATE) FROM {table_name}")
        result = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if result is not None:
            return datetime.strptime(result, "%Y-%m-%d")
        return None
    except Exception as e:
        print(f"❌ MySQL error when checking last GL_DATE: {e}")
        return None

def main():
    # Start from the day after the last GL_DATE in MySQL, or from 2016-01-01 if table is empty
    last_date = get_last_gl_date_mysql(mysql_table)
    if last_date is not None:
        start_date = last_date + timedelta(days=1)
        print(f"Resuming from {start_date.strftime('%Y-%m-%d')} (last in MySQL: {last_date.strftime('%Y-%m-%d')})")
    else:
        start_date = datetime(2002, 2, 26)
        print("No data in MySQL table, starting from 2002-02-26")
    today = datetime.today()
    day = start_date
    first_table_created = False
    while day <= today:
        gl_date_str = day.strftime("%Y-%m-%d")
        print(f"\n=== Processing {gl_date_str} ===")
        df = fetch_oracle_data(gl_date_str)
        if not df.empty:
            print(df.head())
            if not first_table_created:
                ensure_mysql_table_exists(df, mysql_table)
                first_table_created = True
            insert_to_mysql(df, mysql_table)
        else:
            print(f"❌ No data fetched from Oracle for {gl_date_str}. Skipping insert.")
        day += timedelta(days=1)


if __name__ == "__main__":
    main()
