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

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


RME_Material_Movement_query="""(
   SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE ex.DOCUMENT_HEADER_ID = poh.po_header_id AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       NVL (pol.ATTRIBUTE4, poh.ATTRIBUTE4) task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       rsh.RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.name
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       poh.segment1 po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.rcv_transactions rcv,
       APPS.rcv_shipment_headers rsh,
       APPS.po_headers_all poh,
       APPS.po_lines_all pol,
       APPS.rcv_shipment_lines rsl,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.rcv_transaction_id = rcv.transaction_id
       AND mmt.organization_id = hou.organization_id
       AND poh.po_header_id = pol.po_header_id
       AND poh.po_header_id = rsl.po_header_id
       AND rsl.shipment_header_id = rsh.shipment_header_id
       AND rcv.po_header_id = poh.po_header_id
       AND rcv.po_line_id = pol.po_line_id
       AND rcv.shipment_line_id = rsl.shipment_line_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND rsh.shipment_header_id = rcv.shipment_header_id
       AND MMT.transaction_type_id IN (36,
                                       83,
                                       48,
                                       58,
                                       43,
                                       234,
                                       233,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       44,
                                       25,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
-----------------------------------------------------------------------------------------------------     
UNION
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE ex.DOCUMENT_HEADER_ID = poh.po_header_id AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       NVL ( (SELECT PP.TASK_NUMBER
                FROM PA.PA_TASKS PP
               WHERE PP.TASK_ID = MMT.SOURCE_TASK_ID AND ROWNUM = 1),
            poh.ATTRIBUTE4)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       rsh.RECEIPT_NUM,
       (SELECT DISTINCT v.vendor_name
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT v.segment1
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND L.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.name
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       poh.segment1 po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.rcv_transactions rcv,
       APPS.rcv_shipment_headers rsh,
       APPS.po_headers_all poh,
       APPS.po_lines_all pol,
       APPS.rcv_shipment_lines rsl,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.rcv_transaction_id = rcv.transaction_id
       AND mmt.organization_id = hou.organization_id
       AND poh.po_header_id = pol.po_header_id
       AND poh.po_header_id = rsl.po_header_id
       AND rsl.shipment_header_id = rsh.shipment_header_id
       AND rcv.po_header_id = poh.po_header_id
       AND rcv.po_line_id = pol.po_line_id
       AND rcv.shipment_line_id = rsl.shipment_line_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND rsh.shipment_header_id = rcv.shipment_header_id
       AND MMT.transaction_type_id IN (18,
                                       83,
                                       48,
                                       234,
                                       58,
                                       43,
                                       233,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       1003,
                                       44,
                                       25,
                                       71,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
    
       
 ---------------------------------------------------------------------------------------                   
UNION ALL
---nEW aYA-----
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE ex.DOCUMENT_HEADER_ID = poh.po_header_id AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       NVL ( (SELECT PP.TASK_NUMBER
                FROM PA.PA_TASKS PP
               WHERE PP.TASK_ID = MMT.SOURCE_TASK_ID AND ROWNUM = 1),
            poh.ATTRIBUTE4)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       rsh.RECEIPT_NUM,
       (SELECT DISTINCT v.vendor_name
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT v.segment1
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND L.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.NAME
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       poh.segment1 po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.rcv_transactions rcv,
       APPS.rcv_shipment_headers rsh,
       APPS.po_headers_all poh,
       APPS.po_lines_all pol,
       APPS.rcv_shipment_lines rsl,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.rcv_transaction_id = rcv.transaction_id
       AND mmt.organization_id = hou.organization_id
       AND poh.po_header_id = pol.po_header_id
       AND poh.po_header_id = rsl.po_header_id
       AND rsl.shipment_header_id = rsh.shipment_header_id
       AND rcv.po_header_id = poh.po_header_id
       AND rcv.po_line_id = pol.po_line_id
       AND rcv.shipment_line_id = rsl.shipment_line_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND rsh.shipment_header_id = rcv.shipment_header_id
       AND MMT.transaction_type_id IN (83,
                                       48,
                                       58,
                                       43,
                                       234,
                                       233,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       1003,
                                       44,
                                       25,
                                       71,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
     
       AND UPPER (SUBINVENTORY_CODE) = UPPER ('RF1_MAIN')
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
   
     
 ------------------------------------------------------------------------------------------- 
----eND AyA
UNION ALL
---nEW aYA-----
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE ex.DOCUMENT_HEADER_ID = poh.po_header_id AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       NVL ( (SELECT PP.TASK_NUMBER
                FROM PA.PA_TASKS PP
               WHERE PP.TASK_ID = MMT.SOURCE_TASK_ID AND ROWNUM = 1),
            poh.ATTRIBUTE4)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       rsh.RECEIPT_NUM,
       (SELECT DISTINCT v.vendor_name
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT v.segment1
          FROM APPS.po_vendors v
         WHERE v.vendor_id = poh.vendor_id)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND L.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.NAME
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       poh.segment1 po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.rcv_transactions rcv,
       APPS.rcv_shipment_headers rsh,
       APPS.po_headers_all poh,
       APPS.po_lines_all pol,
       APPS.rcv_shipment_lines rsl,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.rcv_transaction_id = rcv.transaction_id
       AND mmt.organization_id = hou.organization_id
       AND poh.po_header_id = pol.po_header_id
       AND poh.po_header_id = rsl.po_header_id
       AND rsl.shipment_header_id = rsh.shipment_header_id
       AND rcv.po_header_id = poh.po_header_id
       AND rcv.po_line_id = pol.po_line_id
       AND rcv.shipment_line_id = rsl.shipment_line_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND rsh.shipment_header_id = rcv.shipment_header_id
       AND MMT.transaction_type_id IN (83,
                                       48,
                                       58,
                                       43,
                                       234,
                                       233,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       1003,
                                       44,
                                       25,
                                       71,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
      
       AND UPPER (SUBINVENTORY_CODE) = UPPER ('RD7_Main')
   --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
---------------------------------------------------------------------------------------------------------------
     
--            ----eND AyA
/*UNION ALL
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT CASE
                  WHEN ms.segment1 LIKE '14%' THEN 'Cement'
                  WHEN ms.segment1 LIKE '15%' THEN 'Steel'
                  WHEN ms.segment1 LIKE '09%' THEN 'Wood'
                  ELSE 'Tools'
               END
          FROM APPS.mtl_system_items_b ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE     ex.expenditure_type = mmt.expenditure_type
               AND mmt.inventory_item_id = ex.inventory_item_id
               AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       (SELECT PP.TASK_NUMBER
          FROM PA.PA_TASKS PP
         WHERE PP.TASK_ID = MMT.SOURCE_TASK_ID AND ROWNUM = 1)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       hou.ORGANIZATION_NAME TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       NULL RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.name
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       NULL po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.organization_id = hou.organization_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND MMT.transaction_type_id IN (186,
                                       206,
                                       120,
                                       140,
                                       83,
                                       234,
                                       48,
                                       58,
                                       43,
                                       233,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       1003,
                                       44,
                                       25,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'*/
------------------------------------------------------------------------------------  
UNION ALL
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE     ex.expenditure_type = mmt.expenditure_type
               AND mmt.inventory_item_id = ex.inventory_item_id
               AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       (SELECT pt.task_number
          FROM APPS.pa_tasks pt
         WHERE pt.task_id = mmt.task_id AND pt.project_id = mmt.project_id)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       NULL RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.name
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       NULL po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
       AND mmt.organization_id = hou.organization_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND MMT.transaction_type_id IN (3,
                                       12,
                                       83,
                                       48,
                                       234,
                                       58,
                                       43,
                                      
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       56,
                                       808,
                                       343,
                                       1003,
                                       44,
                                       25,
                                       889)
       AND mmt.transaction_type_id = ty.transaction_type_id
     
       AND mmt.PRIMARY_QUANTITY >= 0
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
------------------------------------------------------------------------------------       
UNION ALL
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE     ex.expenditure_type = mmt.expenditure_type
               AND mmt.inventory_item_id = ex.inventory_item_id
               AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.ORGANIZATION_ID)
          from_cost_name,
       (SELECT o.ORGANIZATION_NAME
          FROM APPS.org_organization_definitions o
         WHERE o.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       (SELECT pt.task_number
          FROM APPS.pa_tasks pt
         WHERE pt.task_id = mmt.task_id AND pt.project_id = mmt.project_id)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       (SELECT DISTINCT ho.ORGANIZATION_NAME
          FROM APPS.ORG_ORGANIZATION_DEFINITIONS ho
         WHERE HO.ORGANIZATION_ID = mmt.TRANSFER_ORGANIZATION_ID)
          TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       NULL RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       (SELECT pa_all.SEGMENT1
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          PROJECT_NUMBER,
       (SELECT pa_all.name
          FROM APPS.PA_PROJECTS_ALL pa_all
         WHERE pa_all.SEGMENT1 =
                  (SELECT ORG.ATTRIBUTE1
                     FROM APPS.HR_ORGANIZATION_UNITS_V ORG,
                          APPS.ORG_ORGANIZATION_DEFINITIONS2 OOD
                    WHERE     ORG.ORGANIZATION_ID = OOD.ORGANIZATION_ID
                          AND ORG.ORGANIZATION_ID = mmt.ORGANIZATION_ID))
          project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       NULL po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID
     
       AND mmt.organization_id = hou.organization_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND MMT.transaction_type_id IN (3,
                                       21,
                                       83,
                                       48,
                                       58,
                                       32,
                                       91,
                                       57,
                                       82,
                                       17,
                                       90,
                                       55,
                                       
                                       56,
                                       808,
                                       343,
                                       1003,
                                       44,
                                       25,
                                       889,
                                       211)
       AND mmt.transaction_type_id = ty.transaction_type_id
       AND mmt.PRIMARY_QUANTITY < 0
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
 -----------------------------------------------------------------------------------------------  
UNION ALL
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE     ex.expenditure_type = mmt.expenditure_type
               AND mmt.inventory_item_id = ex.inventory_item_id
               AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       NULL from_cost_name,
       NULL to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       (SELECT pt.task_number
          FROM APPS.pa_tasks pt
         WHERE pt.task_id = mmt.task_id AND pt.project_id = mmt.project_id)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       NULL RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       --
       NULL PROJECT_NUMBER,
       NULL project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       NULL po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID(+) = Mmt.INVENTORY_ITEM_ID
       AND mmt.organization_id = hou.organization_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND mmt.TRANSACTION_TYPE_id = 211
       AND mmt.transaction_type_id = ty.transaction_type_id
     --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
 ---------------------------------------------------------------------------------------------    
       UNION ALL
SELECT DISTINCT
       to_char(mmt.TRANSACTION_DATE,'YYYY-MM-DD') TRANSACTION_DATE,
       (SELECT ms.Categor
          FROM APPS.RME_ITEMS_CATEGORY ms
         WHERE ms.INVENTORY_ITEM_ID = Mmt.INVENTORY_ITEM_ID AND ROWNUM = 1)
          Categor,
       (SELECT DISTINCT ex.EXPENDITURE_ITEM_ID
          FROM APPS.pa_expenditure_items_all ex
         WHERE     ex.expenditure_type = mmt.expenditure_type
               AND mmt.inventory_item_id = ex.inventory_item_id
               AND ROWNUM = 1)
          EXPENDITURE_ITEM_ID,
       NULL from_cost_name,
       NULL to_cost_name,
       mmt.transaction_id,
       mmt.SUBINVENTORY_CODE,
       mmt.TRANSACTION_UOM,
       mmt.ACTUAL_COST,
       mmt.PRIMARY_QUANTITY,
       (SELECT pt.task_number
          FROM APPS.pa_tasks pt
         WHERE pt.task_id = mmt.task_id AND pt.project_id = mmt.project_id)
          task_num,
       mmt.expenditure_type expend,
       mmt.CURRENCY_CONVERSION_RATE,
       NULL TRANSFER_SUBINVENTORY,
       msi.SEGMENT1 item_code,
       msi.DESCRIPTION item_desc,
       NULL RECEIPT_NUM,
       NVL ( (SELECT DISTINCT (SELECT DISTINCT v.vendor_name
                                 FROM APPS.po_vendors v
                                WHERE v.vendor_id = h.attribute2)
                FROM APPS.MTL_TXN_REQUEST_HEADERS h
               WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID),
            (SELECT DISTINCT h.attribute3
               FROM APPS.MTL_TXN_REQUEST_HEADERS h
              WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID))
          supplier,
       (SELECT DISTINCT DECODE ( (SELECT DISTINCT v.vendor_name
                                    FROM APPS.po_vendors v
                                   WHERE v.vendor_id = h.attribute2),
                                NULL, '',
                                'supplier')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          suuplier_on,
       (SELECT DISTINCT DECODE (h.attribute3, NULL, '', 'Employee')
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          employee_on,
       (SELECT DISTINCT (SELECT DISTINCT v.segment1
                           FROM APPS.po_vendors v
                          WHERE v.vendor_id = h.attribute2)
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          supplier_no,
       (SELECT DISTINCT l.attribute2
          FROM APPS.MTL_TXN_REQUEST_LINES l
         WHERE     l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.ATTRIBUTE_CATEGORY IN ('Playa', 'Rowad')
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          area,
       (SELECT DISTINCT DECODE (s.attribute2, NULL, 'Employee', 'Supplier')
          FROM APPS.MTL_TXN_REQUEST_LINES l,
               APPS.MTL_TXN_REQUEST_HEADERS h,
               APPS.ap_suppliers s
         WHERE     l.header_id = h.header_id
               AND s.vendor_id = h.attribute2
               AND l.TRANSACTION_HEADER_ID = mmt.TRANSACTION_set_ID
               AND mmt.TRANSACTION_TYPE_ID = l.TRANSACTION_TYPE_ID
               AND l.ORGANIZATION_ID = hou.ORGANIZATION_ID
               AND l.inventory_item_id = mmt.inventory_item_id
               AND mmt.TRX_SOURCE_LINE_ID = l.line_id)
          received_type,
       (SELECT h.REQUEST_NUMBER
          FROM APPS.MTL_TXN_REQUEST_HEADERS h
         WHERE h.header_id = mmt.TRANSACTION_SOURCE_ID)
          MO_NUM,
       --
       NULL PROJECT_NUMBER,
       NULL project,
       mmt.PRIMARY_QUANTITY * mmt.actual_cost total_amount,
       NULL po_number,
       TY.TRANSACTION_TYPE_NAME TRX_TYPE,
       mmt.transaction_set_id,
       mmt.TRANSACTION_TYPE_ID,
       hou.ORGANIZATION_ID,
       msi.inventory_item_id,
       mmt.TRX_SOURCE_LINE_ID
  FROM APPS.MTL_MATERIAL_TRANSACTIONS mmt,
       APPS.mtl_system_items_b msi,
       APPS.ORG_ORGANIZATION_DEFINITIONS hou,
       APPS.mtl_transaction_types TY
 WHERE     msi.INVENTORY_ITEM_ID(+) = Mmt.INVENTORY_ITEM_ID
       AND mmt.organization_id = hou.organization_id
       AND msi.ORGANIZATION_ID = mMt.ORGANIZATION_ID
       AND mmt.TRANSACTION_TYPE_id in(608,548,648,652,649,650,1068)
       AND mmt.transaction_type_id = ty.transaction_type_id
       --AND TO_CHAR(mmt.TRANSACTION_DATE, 'YYYY-MM-DD') BETWEEN '2024-10-01' AND '2024-10-31'
)  temp """

def RME_Material_Movement_ETL():
    spark = fx.spark_app('RME_Material_Movement','10g','4')
    RES = fx.connection(spark,'RES','RMEDB',RME_Material_Movement_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'RME_Material_Movement_Report','overwrite',conn.mysql_username,conn.mysql_password)  


local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('RME_Material_Movement',catchup=False,default_args=default_args,schedule_interval='30 4 * * *',tags=['4'])


RME_Material_MovementTask= PythonOperator(dag=dag,
                task_id = 'RME_Material_Movement',
                python_callable=RME_Material_Movement_ETL) 

RME_Material_MovementTask
