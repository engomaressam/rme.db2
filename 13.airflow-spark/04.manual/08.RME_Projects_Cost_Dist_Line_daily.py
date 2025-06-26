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
mysql_table = "RME_Projects_Cost_Dist_Line_Report_fixed"

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
	rme_dev.pa_utils4.get_unit_of_measure_m(exp_itm.unit_of_measure,
	exp_itm.expenditure_type) UOM,
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

# Get the last GL_DATE in MySQL (as string YYYY-MM-DD)
def get_last_gl_date_mysql():
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        # Only consider dates up to today
        cursor.execute(f"SELECT MAX(GL_DATE) FROM {mysql_table} WHERE GL_DATE <= %s", (datetime.today().strftime('%Y-%m-%d'),))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result and result[0]:
            return result[0]
        else:
            return '2016-01-01'  # Default start
    except Exception as e:
        print(f"❌ MySQL error: {e}")
        return '2016-01-01'

def delete_day_from_mysql(gl_date):
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {mysql_table} WHERE GL_DATE = %s", (gl_date,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Deleted rows for {gl_date} from MySQL.")
    except Exception as e:
        print(f"❌ MySQL delete error: {e}")

def insert_to_mysql(df):
    if df.empty:
        print("❌ No data to insert.")
        return
    # Enforce column order and print for debug
    expected_cols = [
        'EXPENDITURE_ITEM_ID', 'TRANSACTION_SOURCE', 'PROJECT_ID', 'PROJECT_NUM', 'PROJECT_NAME', 'PROJECT_TYPE',
        'TASK_ID', 'TASK_NUM', 'TASK_NAME', 'BU_ID', 'TOP_TASK_ID', 'BL_ID', 'PROJECT', 'FLOOR', 'AREA', 'SECTOR',
        'AREAS', 'EXPENDITURE_ITEM_DATE', 'EXPENDITURE_TYPE', 'EXPENDITURE_ORG_NAME', 'AMOUNT',
        'DR_CODE_COMBINATION_ID', 'CR_CODE_COMBINATION_ID', 'TRANSFERRED_DATE', 'TRANSFER_STATUS_CODE',
        'ACCT_EVENT_ID', 'BATCH_NAME', 'TRANSFER_REJECTION_REASON', 'INVOICE_NUM', 'SUPPLIER_SITE',
        'DOCUMENT_HEADER_ID', 'SYSTEM_REFERENCE2', 'EXPENDITURE_COMMENT', 'VENDOR_NUMBER', 'VENDOR_NAME',
        'GL_DATE', 'IPC_NO', 'CONCAT_SEG_CR', 'CONCAT_SEG_DR', 'PO_NUMBER', 'ITEM_CODE', 'OWNER', 'X1', 'X2',
        'QUANTITY', 'UOM', 'LINE_NUM', 'LINE_DESC'
    ]
    # Only keep columns that are in both expected and DataFrame, in order
    df = df[[col for col in expected_cols if col in df.columns]]
    print("About to insert columns:", list(df.columns))
    try:
        conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        cursor = conn.cursor()
        cols = [f'`{c}`' for c in df.columns]
        insert_sql = f"INSERT INTO {mysql_table} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))})"
        cursor.executemany(insert_sql, df.values.tolist())
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Inserted {len(df)} rows into MySQL.")
    except Exception as e:
        print(f"❌ MySQL insert error: {e}")

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

def main():
    last_gl_date = get_last_gl_date_mysql()
    print(f"Last GL_DATE in MySQL: {last_gl_date}")
    # Ensure last_gl_date is a string before parsing, or use it directly if already a date
    if isinstance(last_gl_date, datetime):
        start_date = last_gl_date
    elif hasattr(last_gl_date, 'strftime'):
        # Handles datetime.date
        start_date = datetime.combine(last_gl_date, datetime.min.time())
    else:
        start_date = datetime.strptime(str(last_gl_date), "%Y-%m-%d")
    today = datetime.today()
    day = start_date
    while day <= today:
        gl_date_str = day.strftime("%Y-%m-%d")
        print(f"\n=== Processing {gl_date_str} ===")
        delete_day_from_mysql(gl_date_str)
        df = fetch_oracle_data(gl_date_str)
        if df.empty:
            print(f"❌ No data fetched from Oracle for {gl_date_str}. Skipping insert.")
        else:
            print(df.head())
            insert_to_mysql(df)
        day += timedelta(days=1)

if __name__ == "__main__":
    main()

"""
# NOTE: You must copy the full SELECT ... JOIN ... FROM ... part from your Airflow DAG query
# and replace the get_oracle_query() function body with the full SQL, parameterizing the date as shown above.
"""
