create view RME_Projects_Cost_Dist_Line as 
(SELECT DISTINCT dis_ln.expenditure_item_id,
                pts.user_transaction_source transaction_source,
                (SELECT DISTINCT tl.attribute1
                   FROM mtl_material_transactions mmt,
                        mtl_txn_request_headers   th,
                        mtl_txn_request_lines     tl
                  WHERE th.header_id = tl.header_id
                    AND th.organization_id = tl.organization_id
                    AND tl.line_id = mmt.move_order_line_id
                    AND prj.PROJECT_ID = exp_itm.PROJECT_ID
                    AND tl.task_id = tsk.task_id
                    AND mmt.Inventory_ITEM_ID = exp_itm.Inventory_ITEM_ID
                    AND exp_itm.orig_transaction_reference =
                        CAST(mmt.transaction_id AS VARCHAR(255))
                    and rownum = 1) Project,
                (SELECT DISTINCT tl.attribute2
                   FROM mtl_material_transactions mmt,
                        mtl_txn_request_headers   th,
                        mtl_txn_request_lines     tl
                  WHERE th.header_id = tl.header_id
                    AND th.organization_id = tl.organization_id
                    AND tl.line_id = mmt.move_order_line_id
                    AND prj.PROJECT_ID = exp_itm.PROJECT_ID
                    AND tl.task_id = tsk.task_id
                    AND mmt.Inventory_ITEM_ID = exp_itm.Inventory_ITEM_ID
                    AND exp_itm.orig_transaction_reference =
                        CAST(mmt.transaction_id AS VARCHAR(255))
                    and rownum = 1) FLOOR,
                (SELECT DISTINCT tl.attribute3
                   FROM mtl_material_transactions mmt,
                        mtl_txn_request_headers   th,
                        mtl_txn_request_lines     tl
                  WHERE th.header_id = tl.header_id
                    AND th.organization_id = tl.organization_id
                    AND tl.line_id = mmt.move_order_line_id
                    AND prj.PROJECT_ID = exp_itm.PROJECT_ID
                    AND tl.task_id = tsk.task_id
                    AND mmt.Inventory_ITEM_ID = exp_itm.Inventory_ITEM_ID
                    AND exp_itm.orig_transaction_reference =
                        CAST(mmt.transaction_id AS VARCHAR(255))
                    and rownum = 1) Area,
                (SELECT Sector
                   FROM APPS.RME_PROJECT_SECTORS A,
                        pa_project_players       pp,
                        per_all_people_f         B
                  WHERE A.PROJECT_ID = prj.PROJECT_ID
                    AND pp.person_id = B.person_id
                    AND pp.project_id = prj.project_id
                    AND pp.project_id = exp_itm.project_id
                    AND ROWNUM = 1
                    AND PP.PROJECT_ROLE_TYPE = '1000'
                    AND SYSDATE BETWEEN b.EFFECTIVE_START_DATE AND
                        b.EFFECTIVE_END_DATE
                    and rownum = 1) SECTOR,
                (SELECT DISTINCT pol.attribute2
                   FROM PO_LINE_LOCATIONS_ALL pol
                  WHERE POL.PO_HEADER_ID = inv.attribute5
                    AND POL.org_id = dis_ln.org_id
                    and rownum = 1) AREAS,
                prj.NAME project_name,
                prj.segment1 project_num,
                tsk.task_name,
                tsk.task_number,
                tsk.work_type_id bu_id,
                tsk.service_type_code bl_id,
                dis_ln.gl_date expenditure_item_date,
                exp_itm.expenditure_type,
                ota_general.get_org_name(NVL(exp_itm.override_to_organization_id,
                                             pa_exp.incurred_by_organization_id)) expenditure_org_name,
                dis_ln.amount,
                dis_ln.project_id,
                dis_ln.task_id,
                dis_ln.dr_code_combination_id,
                dis_ln.cr_code_combination_id,
                dis_ln.transferred_date,
                dis_ln.transfer_status_code,
                dis_ln.acct_event_id,
                dis_ln.batch_name,
                dis_ln.transfer_rejection_reason,
                inv.invoice_num,
                (SELECT VSIT.ADDRESS_LINE1
                   FROM AP_SUPPLIER_SITES_ALL VSIT
                  WHERE VSIT.VENDOR_SITE_ID = INV.VENDOR_SITE_ID
                    and rownum = 1) SUPPLIER_SITE,
                exp_itm.document_header_id,
                dis_ln.system_reference2,
                tsk.top_task_id,
                (SELECT EXPENDITURE_COMMENT
                   FROM pa_expenditure_comments
                  WHERE EXPENDITURE_ITEM_ID = exp_itm.EXPENDITURE_ITEM_ID
                    and rownum = 1) EXPENDITURE_COMMENT,
                (SELECT SEGMENT1
                   FROM PO_VENDORS sub
                  WHERE SUb.vendor_id =
                        NVL(exp_itm.VENDOR_ID, pa_exp.VENDOR_ID)
                    and rownum = 1) vendor_number,
                (SELECT vendor_name
                   FROM PO_VENDORS sub
                  WHERE SUb.vendor_id =
                        NVL(exp_itm.VENDOR_ID, pa_exp.VENDOR_ID)
                    and rownum = 1) VENDOR_name,
                dis_ln.gl_date,
                exp_itm.ORIG_TRANSACTION_REFERENCE IPC_NO,
                NULL concat_seg_cr,
                NULL concat_seg_dr,
                NVL((SELECT po.segment1
                      FROM po_headers_all po
                     WHERE po.po_header_id = exp_itm.document_header_id
                       and rownum = 1),
                    (SELECT po.segment1
                       FROM po_headers_all po
                      WHERE po.po_header_id =
                            NVL(inv.quick_po_header_id, inv.attribute5)
                        and rownum = 1)) po_number,
                (SELECT SEGMENT1
                   FROM mtl_system_items_b
                  WHERE INVENTORY_ITEM_ID = exp_itm.INVENTORY_ITEM_ID
                    AND ROWNUM = 1) ITEM_CODE,
                NVL(NVL((SELECT SEGMENT_VALUE
                          FROM PA_SEGMENT_VALUE_LOOKUPS SS
                         WHERE SEGMENT_VALUE_LOOKUP_SET_ID = 507
                           AND SEGMENT_VALUE_LOOKUP =
                               (SELECT SUBSTR(SEGMENT1, 0, 2)
                                  FROM mtl_system_items_B
                                 WHERE INVENTORY_ITEM_ID =
                                       exp_itm.INVENTORY_ITEM_ID
                                   AND ROWNUM = 1)),
                        (SELECT ASS_ATTRIBUTE1
                           FROM PER_ALL_ASSIGNMENTS_F
                          WHERE SYSDATE BETWEEN EFFECTIVE_START_DATE AND
                                EFFECTIVE_END_DATE
                            AND PERSON_ID = (SELECT POH.AGENT_ID
                                               FROM PO_HEADERS_ALL POH
                                              WHERE POH.PO_HEADER_ID =
                                                    NVL(NVL(inv.quick_po_header_id,
                                                            inv.attribute5),
                                                        exp_itm.document_header_id)
                                                AND ROWNUM = 1))),
                    'Finacial') owner,
                NVL(inv.quick_po_header_id, inv.attribute5) x1,
                exp_itm.document_header_id x2,
                q_exp.Quantity Quantity,
                --          q_exp.UNIT_OF_MEASURE,
                q_exp.UNIT_OF_MEASURE_M UOM,
                NVL((SELECT DISTINCT ss.line_no
                      FROM apps.rme_prj_cont_lines ss,
                           apps.rme_prj_cont_wc    wc
                     WHERE exp_itm.ORIG_TRANSACTION_REFERENCE = wc.ipc_no
                       AND ss.wc_id = wc.wc_id
                       AND prj.project_id = ss.project_id
                       AND ROWNUM = 1),
                    dis_ln.line_num) line_num,
                NVL((SELECT DISTINCT ss.LINE_DESC
                      FROM apps.rme_prj_cont_lines ss,
                           apps.rme_prj_cont_wc    wc
                     WHERE exp_itm.ORIG_TRANSACTION_REFERENCE = wc.ipc_no
                       AND ss.wc_id = wc.wc_id
                       AND prj.project_id = ss.project_id
                       AND ROWNUM = 1),
                    (SELECT EXPENDITURE_COMMENT
                       FROM pa_expenditure_comments
                      WHERE EXPENDITURE_ITEM_ID = exp_itm.EXPENDITURE_ITEM_ID
                        and rownum = 1)) LINE_DESC
--Ended y A.Zaki 21-01-2017
  FROM pa_cost_distribution_lines_all dis_ln,
       pa_projects_all                prj,
       hr_organization_units          org,
       gl_code_combinations           gl,
       pa_tasks                       tsk,
       pa_expenditure_items_all       exp_itm,
       PA_EXPEND_ITEMS_ADJUST2_V      q_exp,
       gl_code_combinations           cr_gl,
       pa_expenditures_all            pa_exp,
       ap_invoices_all                inv,
       pa_transaction_sources         pts
 WHERE tsk.project_id = prj.project_id
   AND dis_ln.project_id = prj.project_id
   AND dis_ln.dr_code_combination_id = gl.code_combination_id(+)
   AND dis_ln.cr_code_combination_id = cr_gl.code_combination_id(+)
   AND q_exp.task_id(+) = tsk.task_id
   AND dis_ln.task_id = tsk.task_id
   AND exp_itm.expenditure_item_id = dis_ln.expenditure_item_id
   AND exp_itm.expenditure_id = pa_exp.expenditure_id
   AND exp_itm.document_header_id = inv.invoice_id(+)
   AND exp_itm.transaction_source = pts.transaction_source(+)
   AND q_exp.expenditure_item_id = dis_ln.expenditure_item_id
   AND q_exp.expenditure_id = pa_exp.expenditure_id
   AND q_exp.document_header_id = inv.invoice_id(+)
   AND q_exp.transaction_source = pts.transaction_source(+)
   AND q_exp.project_id = prj.project_id
   AND prj.carrying_out_organization_id = org.organization_id
   AND dis_ln.amount != 0)
       
       
       --Started By A.Zaki 22-04-2018
