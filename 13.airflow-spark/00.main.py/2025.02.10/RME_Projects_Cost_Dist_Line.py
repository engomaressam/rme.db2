import sys
sys.path.insert(1, '/home/PMO/airflow/dags')
from pyspark.sql import SparkSession
from pyspark import SparkConf, conf
from pyspark.sql.types import *
import pandas as pd
from datetime import * 
import Connections as conn
import ETLFunctions as fx
from time import * 
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import random 
import sys 
import pendulum
import mysql.connector as mysql

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


RME_Projects_Cost_Dist_Line_query="""(
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
	ota_general.get_org_name(NVL(exp_itm.override_to_organization_id,
                                             pa_exp.incurred_by_organization_id)) expenditure_org_name,
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
    and to_char(dis_ln.gl_date, 'YYYY-MM-DD') BETWEEN TO_CHAR(TRUNC(SYSDATE, 'YEAR'), 'YYYY-MM-DD') AND 
      TO_CHAR(TRUNC(SYSDATE) - 1, 'YYYY-MM-DD')
)  temp """

def delete_data():
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = "RME_TEST"
    )
        cursor = db.cursor()
        operation = "DELETE FROM RME_TEST.RME_Projects_Cost_Dist_Line_Report WHERE GL_DATE BETWEEN DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL DAYOFYEAR(CURDATE()) - 1 DAY), '%Y-%m-%d') AND DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d');"
        cursor.execute(operation)
        db.commit()

def stop_spark_on_failure(spark, error_message):
    print(f"Error occurred: {error_message}")
    spark.stop()  # Stop the Spark session
    exit(1)  # Exit with a non-zero status code to signal failure

def Cost_Dist_Line_ETL():
    try:
        # Initialize Spark session
        spark = fx.spark_app('Cost_Dist_Line','8g','4')
        
        # Run the query and handle connection
        RES = fx.connection(spark, 'RES', 'RMEDB', RME_Projects_Cost_Dist_Line_query, 'TEMP', 'ERP')
        
        # Write results to MySQL
        fx.WriteFunction(RES, load_connection_string, 'RME_Projects_Cost_Dist_Line_Report', 'append', conn.mysql_username, conn.mysql_password)
    
    except Exception as e:
        # Catch the exception and stop Spark
        stop_spark_on_failure(spark, str(e))    
        

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,6, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('Cost_Dist_Line',catchup=False,default_args=default_args,schedule_interval='15 0 * * *',tags=['0'])



ReadCost_Dist_LineTask= PythonOperator(dag=dag,
                task_id = 'ReadCost_Dist_LineTask',
                python_callable=Cost_Dist_Line_ETL) 

delete_dataTask= PythonOperator(dag=dag,
                task_id = 'delete_dataTask',
                python_callable=delete_data) 


delete_dataTask >> ReadCost_Dist_LineTask
