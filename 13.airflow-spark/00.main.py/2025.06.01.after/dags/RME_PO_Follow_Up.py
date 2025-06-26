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


PO_Follow_Up_query="""(
   SELECT 
	CASE
		WHEN NVL (d.tax_amt,
		0) = 0
          THEN
             0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
          THEN
             0
		ELSE
               (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
             * (pod.quantity_ordered - NVL (pod.quantity_cancelled,
		0))
	END AS recovery_tax,
	RME_DEV.XXRME_TAX_RATE.GET_TAX_RATE_NAME (poh.po_header_id,
	pll.line_location_id)
          Tax_Code,
	CASE
		WHEN NVL (d.tax_amt,
		0) = 0
            THEN
               0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
            THEN
               0
		ELSE
                 (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
               * (pod.quantity_ordered - NVL (pod.quantity_cancelled,
		0))
	END
       * CASE
		WHEN poh.CURRENCY_CODE = 'EGP' THEN 1
		ELSE poh.rate
	END
          CONVERTED_TAX,
	poh.po_header_id po_header_id,
	pol.po_line_id po_line_id,
	DECODE (poh.style_id,
	1,
	pol.attribute5,
	100,
	pod.expenditure_type)
          exp_type,
	(
	SELECT
		DISTINCT a.expenditure_category expenditure_category
	FROM
		apps.pa_expenditure_types a
	WHERE
		a.expenditure_type =
                      DECODE (poh.style_id,
		1,
		pol.attribute5,
		100,
		pod.expenditure_type)
		AND ROWNUM = 1)
          exp_category,
	DECODE (poh.style_id,
	1,
	NVL (poh.ATTRIBUTE4,
	pol.attribute4),
	100,
	pod.task_id)
          tsk,
	pod.task_id task_id,
	pol.attribute4 task_number,
	(
	SELECT
		DISTINCT tsk1.task_name task_name
	FROM
		apps.pa_tasks tsk1
	WHERE
		pol.attribute4 = tsk1.task_number
		AND ROWNUM = 1)
          tsk_name,
	pll.line_location_id line_location_id,
	pod.po_distribution_id po_distribution_id,
	poh.segment1 po_num,
	poh.comments comments,
	poh.currency_code currency_code,
	(
	SELECT
		tt.name
	FROM
		apps.ap_terms tt
	WHERE
		tt.term_id = poh.terms_id
		AND ROWNUM = 1)
          term,
	(
	SELECT
		tt.description
	FROM
		apps.ap_terms tt
	WHERE
		tt.term_id = poh.terms_id
		AND ROWNUM = 1)
          term_desc,
	to_char(poh.creation_date, 'YYYY-MM-DD') poh_creation_date,
	to_char(poh.creation_date, 'YYYY-MM-DD') poh_crt_dt_line,
	to_char(poh.approved_date, 'YYYY-MM-DD') approved_date,
	to_char(poh.approved_date, 'YYYY-MM-DD') aprv_date_line,
	pol.line_num line_num,
	pol.unit_meas_lookup_code UOM,
	pol.item_description item_description,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	pol.unit_price,
	0,
	( pol.unit_price
              + CASE
		WHEN NVL (d.tax_amt,
		0) = 0 THEN 0
		WHEN NVL (d.trx_line_quantity,
		0) = 0 THEN 0
		ELSE (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
	END),
	( pol.unit_price
           + CASE
		WHEN NVL (d.tax_amt,
		0) = 0 THEN 0
		WHEN NVL (d.trx_line_quantity,
		0) = 0 THEN 0
		ELSE (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
	END))
          unit_price,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	pol.unit_price
               - CASE
		WHEN NVL (d.tax_amt,
		0) = 0 THEN 0
		WHEN NVL (d.trx_line_quantity,
		0) = 0 THEN 0
		ELSE (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
	END,
	0,
	pol.unit_price,
	pol.unit_price)
          unit_price_with_tax,
	pol.attribute1 req_dept,
	to_char(pll.need_by_date, 'YYYY-MM-DD') need_by_date,
	pod.quantity_ordered quantity_ordered,
	pod.quantity_billed quantity_billed,
	pod.quantity_delivered quantity_delivered,
	pod.quantity_cancelled quantity_cancelled,
	pll.quantity_received quantity_received,
	pll.quantity_accepted quantity_accepted,
	pll.quantity_rejected quantity_rejected,
	--led.currency_code currency_code,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                  * pol.unit_price)),
	0,
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                 * pol.unit_price)
              + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                   THEN
                      0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                   THEN
                      0
		ELSE
                        (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                      * ( pod.quantity_ordered
                         - NVL (pod.quantity_cancelled,
		0))
	END),
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
              * pol.unit_price)
           + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                THEN
                   0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                THEN
                   0
		ELSE
                     (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                   * (pod.quantity_ordered - NVL (pod.quantity_cancelled,
		0))
	END))
          line_amount,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                  * pol.unit_price)
               - CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                    THEN
                       0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                    THEN
                       0
		ELSE
                         (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                       * ( pod.quantity_ordered
                          - NVL (pod.quantity_cancelled,
		0))
	END,
	0,
	( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
              * pol.unit_price),
	( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
           * pol.unit_price))
          line_amount_without_Tax,
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
             * pol.unit_price)
          + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
               THEN
                  0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
               THEN
                  0
		ELSE
                    (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                  * (pod.quantity_ordered - NVL (pod.quantity_cancelled,
		0))
	END)
       / pod.quantity_ordered
       * POH.RATE
          AS open_qty_unit_price,
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
             * pol.unit_price)
          + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
               THEN
                  0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
               THEN
                  0
		ELSE
                    (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                  * (pod.quantity_ordered - NVL (pod.quantity_cancelled,
		0))
	END)
       / pod.quantity_ordered
       * POH.RATE
       * ( NVL (pod.quantity_ordered,
	0)
          - NVL (pod.quantity_delivered,
	0)
          - NVL (pod.quantity_cancelled,
	0))
          AS open_qty_amount,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	( ( ( ( pod.quantity_ordered
                        - NVL (pod.quantity_cancelled,
	0))
                     * pol.unit_price))
                * CASE
		WHEN poh.CURRENCY_CODE = 'EGP' THEN 1
		ELSE poh.rate
	END),
	0,
	( ( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                    * pol.unit_price)
                 + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                      THEN
                         0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                      THEN
                         0
		ELSE
                           (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                         * ( pod.quantity_ordered
                            - NVL (pod.quantity_cancelled,
		0))
	END)
              * CASE
		WHEN poh.CURRENCY_CODE = 'EGP' THEN 1
		ELSE poh.rate
	END),
	( ( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                 * pol.unit_price)
              + CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                   THEN
                      0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                   THEN
                      0
		ELSE
                        (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                      * ( pod.quantity_ordered
                         - NVL (pod.quantity_cancelled,
		0))
	END)
           * CASE
		WHEN poh.CURRENCY_CODE = 'EGP' THEN 1
		ELSE poh.rate
	END))
          AS converted_line_amount,
	DECODE (
            RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
               poh.po_header_id,
	pll.line_location_id),
	100,
	( ( ( pod.quantity_ordered
                         - NVL (pod.quantity_cancelled,
	0))
                      * pol.unit_price))
                 - CASE
		WHEN NVL (d.tax_amt,
		0) = 0
                      THEN
                         0
		WHEN NVL (d.trx_line_quantity,
		0) = 0
                      THEN
                         0
		ELSE
                           (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
                         * ( pod.quantity_ordered
                            - NVL (pod.quantity_cancelled,
		0))
	END,
	0,
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
                  * pol.unit_price)),
	( ( (pod.quantity_ordered - NVL (pod.quantity_cancelled,
	0))
               * pol.unit_price)))
       * CASE
		WHEN poh.CURRENCY_CODE = 'EGP' THEN 1
		ELSE poh.rate
	END
          AS converted_amount_without_tax,
	POH.RATE AS current_rate,
	( NVL (pod.quantity_ordered,
	0)
        - NVL (pod.quantity_delivered,
	0)
        - NVL (pod.quantity_cancelled,
	0))
          OPEN,
	(
	SELECT
		DECODE (gcc.segment6,
		'0000',
		gcc.segment5,
		gcc.segment6) project_num
	FROM
		apps.gl_code_combinations_v gcc,
		apps.mtl_parameters mp
	WHERE
		gcc.code_combination_id = mp.material_account
		AND mp.organization_id = pll.ship_to_organization_id
		AND ROWNUM = 1)
          project_num,
	(
	SELECT
		o.organization_name
	FROM
		apps.org_organization_definitions o
	WHERE
		o.organization_id = pll.ship_to_organization_id
		AND ROWNUM = 1)
          project_name,
	(
	SELECT
		o.organization_code organization_code
	FROM
		apps.org_organization_definitions o
	WHERE
		o.organization_id = pll.ship_to_organization_id
		AND ROWNUM = 1)
          organization_code,
	(
	SELECT
		msi.segment1 segment1
	FROM
		apps.mtl_system_items_b msi
	WHERE
		msi.inventory_item_id = pol.item_id
		AND msi.organization_id = pll.ship_to_organization_id
		AND ROWNUM = 1)
          item_code,
	(
	SELECT
		prh.segment1 segment1
	FROM
		apps.po_requisition_headers_all prh,
		apps.po_requisition_lines_all prl,
		apps.po_req_distributions_all prd
	WHERE
		prh.requisition_header_id = prl.requisition_header_id
		AND prl.requisition_line_id = prd.requisition_line_id
		AND prd.distribution_id = pod.req_distribution_id
		AND ROWNUM = 1)
          pr_num,
	(
	SELECT
		prl.line_num line_num
	FROM
		apps.po_requisition_lines_all prl,
		apps.po_req_distributions_all prd
	WHERE
		prl.requisition_line_id = prd.requisition_line_id
		AND prd.distribution_id = pod.req_distribution_id
		AND ROWNUM = 1)
          pr_line_num,
	(
	SELECT
		pov.vendor_name vendor_name
	FROM
		apps.po_vendors pov
	WHERE
		poh.vendor_id = pov.vendor_id
		AND ROWNUM = 1)
          vendor_name,
	(
	SELECT
		pov.segment1 segment1
	FROM
		apps.po_vendors pov
	WHERE
		poh.vendor_id = pov.vendor_id
		AND ROWNUM = 1)
          vendor_no,
	(
	SELECT
		p.full_name full_name
	FROM
		apps.per_all_people_f p
	WHERE
		p.person_id = poh.agent_id
		AND (SYSDATE BETWEEN p.effective_start_date
                                AND p.effective_end_date)
			AND ROWNUM = 1)
          buyer_name,
	(
	SELECT
		prh.attribute3 attribute3
	FROM
		apps.po_requisition_headers_all prh,
		apps.po_requisition_lines_all prl,
		apps.po_req_distributions_all prd
	WHERE
		prh.requisition_header_id = prl.requisition_header_id
		AND prl.requisition_line_id = prd.requisition_line_id
		AND prd.distribution_id = pod.req_distribution_id
		--and pod.PO_LINE_ID = i.po_line_id
		AND ROWNUM = 1)
          pr_category,
	(
	SELECT
		prh.attribute5 attribute5
	FROM
		apps.po_requisition_headers_all prh,
		apps.po_requisition_lines_all prl,
		apps.po_req_distributions_all prd
	WHERE
		prh.requisition_header_id = prl.requisition_header_id
		AND prl.requisition_line_id = prd.requisition_line_id
		AND prd.distribution_id = pod.req_distribution_id
		--and pod.PO_LINE_ID = i.po_line_id
		AND ROWNUM = 1)
          pr_reason,
	poh.authorization_status authorization_status,
	to_char(pll.promised_date, 'YYYY-MM-DD') promised_date,
	(
	SELECT
		l.long_text long_text
	FROM
		apps.fnd_attached_docs_form_vl d,
		apps.fnd_documents_long_text l
	WHERE
		d.media_id = l.media_id
		AND d.function_name = DECODE (0,
		1,
		NULL,
		'PO_POXPOVPO')
			AND d.function_type = DECODE (0,
			1,
			NULL,
			'F')
				AND (d.entity_name = 'PO_HEADERS')
					AND category_id = 1000712
					AND ROWNUM = 1
					AND d.pk1_value =
                      (
					SELECT
						po_header_id
					FROM
						apps.po_headers_all p4
					WHERE
						p4.po_header_id = poh.po_header_id
						AND type_lookup_code = 'STANDARD')
					AND ROWNUM = 1)
          delv,
	CASE
		WHEN pll.cancel_flag = 'Y'
          THEN
             'Cancelled'
		WHEN NVL (pod.quantity_cancelled,
		0) > 0
		AND NVL (pod.quantity_cancelled,
		0) < pod.quantity_ordered
          THEN
             'Partially Cancelled'
		ELSE
             NULL
	END
          AS shipment_cancel_status,
	pll.closed_code AS shipment_close_status,
	(
	SELECT
		per_all.ASS_ATTRIBUTE1 ASS_ATTRIBUTE1
	FROM
		apps.PER_ALL_ASSIGNMENTS_F per_all
	WHERE
		per_all.PERSON_ID = POH.AGENT_ID
		AND SYSDATE BETWEEN per_all.EFFECTIVE_START_DATE
                               AND per_all.EFFECTIVE_END_DATE
		AND ROWNUM = 1)
          BUYER_DEP,
	DECODE (
          RME_DEV.XXRME_TAX_RATE.GET_REC_PERCENTAGE_RATE (
             poh.po_header_id,
	pll.line_location_id),
	100,
	pol.unit_price,
	0,
	( pol.unit_price
              + CASE
		WHEN NVL (d.tax_amt,
		0) = 0 THEN 0
		WHEN NVL (d.trx_line_quantity,
		0) = 0 THEN 0
		ELSE (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
	END),
	( pol.unit_price
           + CASE
		WHEN NVL (d.tax_amt,
		0) = 0 THEN 0
		WHEN NVL (d.trx_line_quantity,
		0) = 0 THEN 0
		ELSE (NVL (d.tax_amt,
		0) / NVL (d.trx_line_quantity,
		0))
	END))*( NVL (pod.quantity_ordered,
	0)
        - NVL (pod.quantity_delivered,
	0)
        - NVL (pod.quantity_cancelled,
	0)) "Quantity Open Amount"
FROM
	apps.po_headers_all poh,
	apps.po_lines_all pol,
	apps.po_line_locations_all pll,
	apps.po_distributions_all pod,
	/*(
	SELECT
		hr_org.organization_id organization_id,
		hr_org.org_information3 AS ledger_id
	FROM
		apps.hr_organization_information hr_org
	WHERE
		hr_org.org_information_context = 'Operating Unit Information') org_led,*/
	--apps.gl_ledgers led,
	(
	SELECT
		SUM (zx.tax_amt) AS tax_amt,
		SUM (zx.trx_line_quantity) AS trx_line_quantity,
		trx_id,
		trx_line_id
	FROM
		apps.zx_lines zx
	WHERE
		zx.entity_code = 'PURCHASE_ORDER'
		AND zx.application_id = 201
		AND zx.event_class_code = 'PO_PA'
		AND zx.trx_level_type = 'SHIPMENT'
		AND zx.tax_apportionment_line_number = 1
	GROUP BY
		zx.trx_id,
		zx.trx_line_id) d
WHERE
	poh.po_header_id = pol.po_header_id
	--AND poh.org_id = org_led.organization_id
	--AND org_led.ledger_id = led.ledger_id
	AND pol.po_line_id = pll.po_line_id
       /*LINK WITH TAX LINES*/
	AND poh.po_header_id = d.trx_id(+)
	AND pll.line_location_id = d.trx_line_id(+)
       /*LINK WITH TAX LINES*/
	AND pol.po_line_id = pod.po_line_id
	AND pll.line_location_id = pod.line_location_id
	AND poh.type_lookup_code = 'STANDARD'
    and to_char(poh.creation_date, 'YYYY-MM-DD') BETWEEN TO_CHAR(TRUNC(SYSDATE, 'YEAR'), 'YYYY-MM-DD') AND 
      TO_CHAR(TRUNC(SYSDATE) - 1, 'YYYY-MM-DD')
)  temp """

def delete_data():
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = "RME_TEST"
    )
        cursor = db.cursor()
        operation = "DELETE FROM RME_TEST.RME_PO_Follow_Up_Report WHERE poh_creation_date BETWEEN DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL DAYOFYEAR(CURDATE()) - 1 DAY), '%Y-%m-%d') AND DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d');"
        cursor.execute(operation)
        db.commit()

def PO_Follow_Up_ETL():
    spark = fx.spark_app('RME_PO_Follow_Up','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',PO_Follow_Up_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'RME_PO_Follow_Up_Report','append',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('PO_Follow_Up',catchup=False,default_args=default_args,schedule_interval='00 4 * * *',tags=['2'])


PO_Follow_UpTask= PythonOperator(dag=dag,
                task_id = 'PO_Follow_Up',
                python_callable=PO_Follow_Up_ETL) 

delete_dataTask= PythonOperator(dag=dag,
                task_id = 'delete_dataTask',
                python_callable=delete_data) 


delete_dataTask >> PO_Follow_UpTask

