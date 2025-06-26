create or replace view RME_Suppliers_Balance_dist    as 
(SELECT DISTINCT org_id,
                attribute10,
                (SELECT DISTINCT full_name
                   FROM po.po_agents        fu,
                        hr.per_all_people_f papf,
                        po_headers_all      poh
                  WHERE fu.agent_id = poh.agent_id
                    AND papf.person_id = fu.agent_id
                    AND ROWNUM = 1
                    AND SYSDATE BETWEEN papf.effective_start_date AND
                        papf.effective_end_date
                    AND poh.po_header_id = quick_po_header_id) buyer,
                (SELECT DISTINCT paf.ass_attribute1
                   FROM po.po_agents      fu,
                        po_headers_all    poh,
                        per_all_people_f  ppf,
                        per_assignments_f paf
                  WHERE fu.agent_id = poh.agent_id
                    AND ppf.person_id = paf.person_id
                    AND fu.agent_id = ppf.person_id
                    AND SYSDATE BETWEEN ppf.effective_start_date AND
                        ppf.effective_end_date
                    AND ROWNUM = 1
                    AND SYSDATE BETWEEN paf.effective_start_date AND
                        paf.effective_end_date
                    AND poh.po_header_id = NVL(quick_po_header_id, dff_po)) buyer_dep,
                (SELECT DISTINCT segment1
                   FROM po_headers_all
                  WHERE po_header_id = NVL(quick_po_header_id, dff_po)) po_num,
                inv_type inv_type,
                inv_type_dtl,
                vendor_id,
                project_name,
                SECTOR,
                supplier_num,
                vendor_name,
                invoice_id,
                invoice_num,
                invoice_amount,
                voucher_number,
                invoice_type,
                currency,
                description,
                gl_date,
                item_total amount,
                amount_paid,
                NVL(item_total, 0) - NVL(amount_paid, 0) net,
                bank_name,
                bank_account,
                bic,
                (SELECT MAX(aip.accounting_date)
                   FROM ap_invoice_payments_all aip,
                        ap_invoices_all         ai1,
                        ap_invoices_all         ai2,
                        ap_checks_all           ac,
                        ap_lookup_codes         alc1,
                        iby_payment_methods_vl  iby
                  WHERE aip.invoice_id = ai1.invoice_id
                    AND aip.other_invoice_id = ai2.invoice_id(+)
                    AND aip.check_id = ac.check_id
                    AND alc1.lookup_type(+) = 'NLS TRANSLATION'
                    AND alc1.lookup_code(+) = 'PREPAY'
                    AND iby.payment_method_code(+) = ac.payment_method_code
                    AND ac.status_lookup_code <> 'VOIDED'
                       --AND AIP.INVOICE_ID = ai.INVOICE_ID
                    AND ai1.vendor_id = main_q.vendor_id ) last_paym_date,
                tot_payment
  FROM (WITH q_ap_invoices AS (SELECT DISTINCT --'Q1' Q,
                                                main_ai.attribute2 inv_type,
                                               main_ai.attribute10,
                                               (SELECT ppa.name
                                                  FROM pa_projects_all ppa
                                                 WHERE ppa.project_id =
                                                       main_ai.attribute6) || '/' ||
                                               DECODE(main_ai.attribute7,
                                                      main_ai.attribute7,
                                                      (SELECT description cost_center_desc
                                                         FROM fnd_flex_values_vl
                                                        WHERE flex_value_set_id =
                                                              1017259
                                                          AND flex_value =
                                                              main_ai.attribute7)) project_name,
                                               (SELECT DISTINCT SECTOR
                                                  FROM APPS.RME_PROJECT_SECTORS A,
                                                       PA_PROJECTS_ALL          PAP
                                                 WHERE 1 = 1 --PH.TRX_ID = RIH.TRX_ID
                                                   AND A.project_id =
                                                       PAP.project_id
                                                   AND PAP.project_id =
                                                       main_ai.attribute6
                                                   AND ROWNUM = 1) SECTOR,
                                               main_ai.org_id,
                                               main_ai.quick_po_header_id,
                                               main_ai.vendor_id,
                                               main_ai.attribute5 dff_po,
                                               aps.segment1 supplier_num,
                                               aps.vendor_name,
                                               main_ai.invoice_id,
                                               main_ai.invoice_num,
                                               main_ai.invoice_amount,
                                               main_ai.doc_sequence_value voucher_number,
                                               main_ai.invoice_type_lookup_code invoice_type,
                                               main_ai.invoice_currency_code currency,
                                               main_ai.description,
                                               main_ai.gl_date,
                                               main_ai.invoice_date,
                                               x.bank_name bank_name,
                                               x.account_number bank_account,
                                               x.swift_code bic,
                                               NULL last_paym_date,
                                               (SELECT SUM(ap.amount) xamt
                                                  FROM apps.ap_invoice_payments_all ap,
                                                       ap_invoices_all              ai
                                                 WHERE ap.invoice_id =  ai.invoice_id
                                                  
                                                 GROUP BY ai.vendor_id,   ap.accounting_date
                                                HAVING(ap.accounting_date) = (SELECT MAX(aip.accounting_date)
                                                                               FROM ap_invoice_payments_all aip,
                                                                                    ap_invoices_all         inner_ai,
                                                                                    ap_suppliers            aps
                                                                              WHERE 1 = 1
                                                                                AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE,
                                                                                        'X') NOT IN
                                                                                    ('EMPLOYEE',
                                                                                     'EMPLOYEE.')
                                                                                AND aip.invoice_id =
                                                                                    inner_ai.invoice_id
                                                                                AND inner_ai.vendor_id =
                                                                                    aps.vendor_id
                                                                                )) tot_payment
                                 FROM ap_invoices_all              main_ai,
                                      ap_invoice_lines_all         ail,
                                      ap_invoice_distributions_all apd,
                                      gl_code_combinations         gcc,
                                      ap_suppliers                 aps,
                                      --
                                      apps.rme_ebanking_details x
                               --
                                WHERE main_ai.invoice_id = ail.invoice_id
                                  AND apd.invoice_id = main_ai.invoice_id
                                  AND apd.invoice_line_number =
                                      ail.line_number
                                  AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE, 'X') NOT IN
                                      ('EMPLOYEE', 'EMPLOYEE.')
                                  AND gcc.code_combination_id =
                                      apd.dist_code_combination_id
                                  AND aps.vendor_id = main_ai.vendor_id
                                  AND x.vendor_id(+) = aps.vendor_id
                                  AND DECODE(x.vendor_id,
                                             NULL,
                                             1,
                                             x.bank_account_pirority) = 1
                                     ----------------------------
                                     --Start User Parameters
                                  
                               UNION
                               SELECT DISTINCT --'Q1' Q,
                                                main_ai.attribute2 inv_type,
                                               main_ai.attribute10,
                                               (SELECT ppa.name
                                                  FROM pa_projects_all ppa
                                                 WHERE ppa.project_id =
                                                       main_ai.attribute6) || '/' ||
                                               DECODE(main_ai.attribute7,
                                                      main_ai.attribute7,
                                                      (SELECT description cost_center_desc
                                                         FROM fnd_flex_values_vl
                                                        WHERE flex_value_set_id =
                                                              1017259
                                                          AND flex_value =
                                                              main_ai.attribute7)) project_name,
                                               (SELECT DISTINCT SECTOR
                                                  FROM APPS.RME_PROJECT_SECTORS A,
                                                       PA_PROJECTS_ALL          PAP
                                                 WHERE 1 = 1 --PH.TRX_ID = RIH.TRX_ID
                                                   AND A.project_id =
                                                       PAP.project_id
                                                   AND PAP.project_id =
                                                       main_ai.attribute6
                                                   AND ROWNUM = 1) SECTOR,
                                               main_ai.org_id,
                                               main_ai.quick_po_header_id,
                                               main_ai.vendor_id,
                                               main_ai.attribute5 dff_po,
                                               aps.segment1 supplier_num,
                                               aps.vendor_name,
                                               main_ai.invoice_id,
                                               main_ai.invoice_num,
                                               main_ai.invoice_amount,
                                               main_ai.doc_sequence_value voucher_number,
                                               main_ai.invoice_type_lookup_code invoice_type,
                                               main_ai.invoice_currency_code currency,
                                               main_ai.description,
                                               main_ai.gl_date,
                                               main_ai.invoice_date,
                                               x.bank_name bank_name,
                                               x.account_number bank_account,
                                               x.swift_code bic,
                                               NULL last_paym_date,
                                               (SELECT SUM(ap.amount) xamt
                                                  FROM apps.ap_invoice_payments_all ap,
                                                       ap_invoices_all              ai
                                                 WHERE ap.invoice_id =      ai.invoice_id
                                                 GROUP BY ai.vendor_id,
                                                          ap.accounting_date
                                                HAVING(ap.accounting_date) = (SELECT MAX(aip.accounting_date)
                                                                               FROM ap_invoice_payments_all aip,
                                                                                    ap_invoices_all         inner_ai,
                                                                                    ap_suppliers            aps
                                                                              WHERE 1 = 1
                                                                                AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE,
                                                                                        'X') NOT IN
                                                                                    ('EMPLOYEE',
                                                                                     'EMPLOYEE.')
                                                                                AND aip.invoice_id =
                                                                                    inner_ai.invoice_id
                                                                                AND inner_ai.vendor_id =
                                                                                    aps.vendor_id
                                                                                AND aps.vendor_id =
                                                                                    ai.vendor_id
                                                                                )) tot_payment
                                 FROM ap_invoices_all              main_ai,
                                      ap_invoice_lines_all         ail,
                                      ap_invoice_distributions_all apd,
                                      gl_code_combinations         gcc,
                                      ap_suppliers                 aps,
                                      --
                                      apps.rme_ebanking_details x
                               --
                                WHERE main_ai.invoice_id = ail.invoice_id
                                  AND apd.invoice_id = main_ai.invoice_id
                                  AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE, 'X') NOT IN
                                      ('EMPLOYEE', 'EMPLOYEE.')
                                  AND apd.invoice_line_number =
                                      ail.line_number
                                  AND gcc.code_combination_id =
                                      apd.dist_code_combination_id
                                  AND aps.vendor_id = main_ai.vendor_id
                                  AND x.vendor_id(+) = aps.vendor_id
                                  AND DECODE(x.vendor_id,
                                             NULL,
                                             1,
                                             x.bank_account_pirority) = 1
                                  AND main_ai.attribute10 NOT IN
                                      ('RME Zero Invoices')
                                     --Start User Parameters
                                  UNION
                               --Query retrive invoice not in period and payment in period
                               SELECT DISTINCT ai.attribute2 inv_type,
                                               ai.attribute10,
                                               (SELECT ppa.name
                                                  FROM pa_projects_all ppa
                                                 WHERE ppa.project_id =
                                                       ai.attribute6) || '/' ||
                                               DECODE(ai.attribute7,
                                                      ai.attribute7,
                                                      (SELECT description cost_center_desc
                                                         FROM fnd_flex_values_vl
                                                        WHERE flex_value_set_id =
                                                              1017259
                                                          AND flex_value =
                                                              ai.attribute7)) project_name,
                                               (SELECT DISTINCT SECTOR
                                                  FROM APPS.RME_PROJECT_SECTORS A,
                                                       PA_PROJECTS_ALL          PAP
                                                 WHERE 1 = 1 --PH.TRX_ID = RIH.TRX_ID
                                                   AND A.project_id =
                                                       PAP.project_id
                                                   AND PAP.project_id =
                                                       AI.attribute6
                                                   AND ROWNUM = 1) SECTOR,
                                               ai.org_id,
                                               ai.quick_po_header_id,
                                               ai.vendor_id,
                                               ai.attribute5 dff_po,
                                               aps.segment1 supplier_num,
                                               aps.vendor_name,
                                               ai.invoice_id,
                                               ai.invoice_num,
                                               ai.invoice_amount,
                                               ai.doc_sequence_value voucher_number,
                                               ai.invoice_type_lookup_code invoice_type,
                                               ai.invoice_currency_code currency,
                                               ai.description,
                                               ai.gl_date,
                                               ai.invoice_date,
                                               x.bank_name bank_name,
                                               x.account_number bank_account,
                                               x.swift_code bic,
                                               NULL last_paym_date,
                                               (SELECT SUM(ap.amount) xamt
                                                  FROM apps.ap_invoice_payments_all ap,
                                                       ap_invoices_all              ai
                                                 WHERE ap.invoice_id =
                                                       ai.invoice_id
                                                 GROUP BY ai.vendor_id,
                                                          ap.accounting_date
                                                HAVING(ap.accounting_date) = (SELECT MAX(aip.accounting_date)
                                                                               FROM ap_invoice_payments_all aip,
                                                                                    ap_invoices_all         inner_ai,
                                                                                    ap_suppliers            aps
                                                                              WHERE 1 = 1
                                                                                AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE,
                                                                                        'X') NOT IN
                                                                                    ('EMPLOYEE',
                                                                                     'EMPLOYEE.')
                                                                                AND aip.invoice_id =
                                                                                    inner_ai.invoice_id
                                                                                AND inner_ai.vendor_id =
                                                                                    aps.vendor_id
                                                                                AND aps.vendor_id =
                                                                                    ai.vendor_id
                                                                                 )) tot_payment
                                 FROM ap_invoice_payments_all      aip,
                                      ap_invoices_v                ai,
                                      ap_suppliers                 aps,
                                      gl_code_combinations         gcc,
                                      ap_invoice_distributions_all apd,
                                      apps.rme_ebanking_details    x
                                WHERE ai.invoice_id = aip.invoice_id
                                  AND gcc.code_combination_id =
                                      apd.dist_code_combination_id
                                  AND apd.invoice_id = ai.invoice_id
                                  AND ai.vendor_id = aps.vendor_id
                                  AND x.vendor_id(+) = aps.vendor_id
                                  AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE, 'X') NOT IN
                                      ('EMPLOYEE', 'EMPLOYEE.')
                                  AND DECODE(x.vendor_id,
                                             NULL,
                                             1,
                                             x.bank_account_pirority) = 1
                                  AND ai.attribute10 NOT IN
                                      ('RME Zero Invoices')
                                  UNION
                               --Query retrive invoice not in period and amount applied in period
                               SELECT DISTINCT ai.attribute2 inv_type,
                                               ai.attribute10,
                                               (SELECT ppa.name
                                                  FROM pa_projects_all ppa
                                                 WHERE ppa.project_id =
                                                       ai.attribute6) || '/' ||
                                               DECODE(ai.attribute7,
                                                      ai.attribute7,
                                                      (SELECT description cost_center_desc
                                                         FROM fnd_flex_values_vl
                                                        WHERE flex_value_set_id =
                                                              1017259
                                                          AND flex_value =
                                                              ai.attribute7)) project_name,
                                               (SELECT DISTINCT SECTOR
                                                  FROM APPS.RME_PROJECT_SECTORS A,
                                                       PA_PROJECTS_ALL          PAP
                                                 WHERE 1 = 1 --PH.TRX_ID = RIH.TRX_ID
                                                   AND A.project_id =
                                                       PAP.project_id
                                                   AND PAP.project_id =
                                                       AI.attribute6
                                                   AND ROWNUM = 1) SECTOR,
                                               ai.org_id,
                                               ai.quick_po_header_id,
                                               ai.vendor_id,
                                               ai.attribute5 dff_po,
                                               aps.segment1 supplier_num,
                                               aps.vendor_name,
                                               ai.invoice_id,
                                               ai.invoice_num,
                                               ai.invoice_amount,
                                               ai.doc_sequence_value voucher_number,
                                               ai.invoice_type_lookup_code invoice_type,
                                               ai.invoice_currency_code currency,
                                               ai.description,
                                               ai.gl_date,
                                               ai.invoice_date,
                                               x.bank_name bank_name,
                                               x.account_number bank_account,
                                               x.swift_code bic,
                                               NULL last_paym_date,
                                               (SELECT NVL(SUM(amount), 0) *
                                                       NVL(inner_inv.exchange_rate,
                                                           1)
                                                  FROM ap_invoice_payment_history_v aph,
                                                       ap_invoices_all              inner_inv
                                                 WHERE aph.invoice_id =
                                                       inner_inv.invoice_id
                                                   AND inner_inv.invoice_id =
                                                       ai.invoice_id
                                                 GROUP BY inner_inv.exchange_rate
                                                HAVING MAX(accounting_date) = (SELECT MAX(aip.accounting_date)
                                                                                FROM ap_invoice_payments_all aip,
                                                                                     ap_invoices_all         ai1,
                                                                                     ap_invoices_all         ai2,
                                                                                     ap_checks_all           ac,
                                                                                     ap_lookup_codes         alc1,
                                                                                     iby_payment_methods_vl  iby
                                                                               WHERE aip.invoice_id =
                                                                                     ai1.invoice_id
                                                                                 AND aip.other_invoice_id =
                                                                                     ai2.invoice_id(+)
                                                                                 AND aip.check_id =
                                                                                     ac.check_id
                                                                                 AND alc1.lookup_type(+) =
                                                                                     'NLS TRANSLATION'
                                                                                 AND alc1.lookup_code(+) =
                                                                                     'PREPAY'
                                                                                 AND iby.payment_method_code(+) =
                                                                                     ac.payment_method_code
                                                                                 AND ac.status_lookup_code <>
                                                                                     'VOIDED'
                                                                                 AND ai1.vendor_id =  aps.vendor_id
                                                                                 )) tot_payment
                                 FROM ap_invoices_all              ai,
                                      ap_invoice_lines_all         ail,
                                      ap_invoice_distributions_all apd,
                                      gl_code_combinations         gcc,
                                      ap_suppliers                 aps,
                                      apps.rme_ebanking_details    x
                                WHERE ai.invoice_id = ail.invoice_id
                                  AND aps.vendor_id = ai.vendor_id
                                  AND NVL(aps.VENDOR_TYPE_LOOKUP_CODE, 'X') NOT IN
                                      ('EMPLOYEE', 'EMPLOYEE.')
                                  AND apd.invoice_id = ai.invoice_id
                                  AND apd.invoice_line_number =
                                      ail.line_number
                                  AND gcc.code_combination_id =
                                      apd.dist_code_combination_id
                                  AND x.vendor_id(+) = aps.vendor_id
                                  AND DECODE(x.vendor_id,
                                             NULL,
                                             1,
                                             x.bank_account_pirority) = 1
                                  AND (apd.line_type_lookup_code = 'PREPAY')
                                     ------AND ai.invoice_type_lookup_code IN ('STANDARD','PREPAYMENT')
                                  AND ai.attribute10 NOT IN
                                      ('RME Zero Invoices'))
       -----------------
       ---Start Query---
       -----------------
         SELECT DISTINCT q_inv.attribute10,
                         --  DECODE(:p_group_invoice_type, 'Y',q_inv.inv_type,'ALL INVOICES') inv_type, --Q,
                         q_inv.inv_type  inv_type,
                         q_inv.inv_type inv_type_dtl,
                         q_inv.project_name,
                         q_inv.SECTOR,
                         q_inv.quick_po_header_id,
                         q_inv.org_id,
                         q_inv.vendor_id,
                         q_inv.dff_po,
                         q_inv.supplier_num,
                         q_inv.vendor_name,
                         q_inv.invoice_id,
                         q_inv.invoice_num,
                         q_inv.invoice_amount,
                         q_inv.voucher_number,
                         q_inv.invoice_type,
                         q_inv.currency,
                         q_inv.description,
                         q_inv.gl_date,
                         q_inv.invoice_date,
                         q_inv.bank_name,
                         q_inv.bank_account,
                         q_inv.bic,
                         q_inv.last_paym_date,
                         q_inv.tot_payment,
                         ------------------------------------------------
                         --==============================================
                         
                         (SELECT SUM(apd.amount *                              NVL(ai.exchange_rate, 1)) item_total
                            FROM ap_invoices_all              ai,
                                 ap_invoice_lines_all         ail,
                                 ap_invoice_distributions_all apd,
                                 gl_code_combinations         gcc
                           WHERE ai.invoice_id = ail.invoice_id
                                ----------------------------
                             AND apd.invoice_id = ai.invoice_id
                             AND apd.invoice_line_number = ail.line_number
                             AND gcc.code_combination_id =
                                 apd.dist_code_combination_id
                                ----------------------------
                             AND ai.vendor_id = q_inv.vendor_id
                             AND ai.invoice_id = q_inv.invoice_id
                                --============================== start user parameter ===========
                             ) item_total,
                         (SELECT SUM(aip.amount *  NVL(ai.exchange_rate, 1))
                            FROM ap_invoice_payments_all aip,
                                 ap_invoices_v           ai
                           WHERE ai.invoice_id = aip.invoice_id
                             AND ai.vendor_id = q_inv.vendor_id
                             AND ai.invoice_id = q_inv.invoice_id
                                --============================== start user parameter ===========
                             ) amount_paid
         --==============================================
         ------------------------------------------------
         
           FROM Q_AP_INVOICES Q_INV
          WHERE VENDOR_ID NOT IN (3192, 19192, 52239, 52240)) MAIN_Q ) 
          
