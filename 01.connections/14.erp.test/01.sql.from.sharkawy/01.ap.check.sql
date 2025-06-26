SELECT ACA.CHECK_NUMBER "Document Number",
         ai.invoice_num,
         (SELECT DISTINCT xel.CODE_COMBINATION_ID
            FROM AP_CHECKS_all ac,
                 xla_events xe,
                 xla_ae_headers xeh,
                 xla_ae_lines xel,
                 xla.xla_transaction_entities xte
           WHERE     xte.application_id = xe.application_id
                 AND ac.CHECK_ID = aca.CHECK_ID
                 AND xte.entity_code = 'AP_PAYMENTS'
                 AND xte.source_id_int_1 = ac.CHECK_ID
                 AND xte.entity_id = xe.entity_id
                 AND xte.application_id = xeh.application_id
                 AND xte.entity_id = xeh.entity_id
                 AND xel.application_id = xeh.application_id
                 AND xel.ae_header_id = xeh.ae_header_id
                 AND ROWNUM = 1)
            comp_id,
         ai.attribute7,
         (SELECT DESCRIPTION
            FROM FND_FLEX_VALUES_TL T, FND_FLEX_VALUES B
           WHERE     B.FLEX_VALUE_ID = T.FLEX_VALUE_ID
                 AND FLEX_VALUE_SET_ID = 1017259
                 AND FLEX_VALUE = ai.attribute7
                 And Description is null)
            cost_center,
         ai.invoice_id,
         aip.amount PAYMENT_AMOUNT,
         aip.amount * NVL (aca.EXCHANGE_RATE, 1) equiv,
         pap.name project,
         NVL (
            (SELECT DISTINCT full_name
               FROM pa_project_players pp, per_all_people_f a
              WHERE     pp.person_id = a.person_id
                    AND pp.project_id = pap.project_id
                    AND ROWNUM = 1
                    AND PROJECT_ROLE_TYPE = '1000'),
            'General')
            owner,
            (SELECT DISTINCT SECTOR
                       FROM RME_PROJECT_SECTORS A,
                            pa_project_players pp,
                            per_all_people_f a
                      WHERE     1 = 1                 --PH.TRX_ID = RIH.TRX_ID
                            AND PAP.PROJECT_ID = A.PROJECT_ID
                            AND pp.person_id = a.person_id
                            AND pp.project_id = pap.project_id
                            AND ROWNUM = 1)SECTOR,
ASA.SEGMENT1 "Supplier Number",
  ASA.VENDOR_NAME "Supplier Name",
         ASA.SEGMENT1 "Supplier Number",
         --   ASA.SEGMENT1 "Supplier Num" ,
         aca.check_id,
         aca.CE_BANK_ACCT_USE_ID,
         cbac.BANK_ACCOUNT_ID,
         ASA.VENDOR_NAME "Supplier Name",
         -- ASA.VENDOR_NAME "Sup Name",
         aca.VENDOR_ID,
         --    aca.BANK_ACCOUNT_ID ,
         aca.PAYCARD_AUTHORIZATION_NUMBER,
         aca.BANK_ACCOUNT_NAME,
         ACA.STATUS_LOOKUP_CODE "Payment Reconcilation Status",
         ACA.CLEARED_AMOUNT "Cleared Amount",
         ACA.CURRENCY_CODE "Currency",
         ACA.CLEARED_DATE "Cleared Date",
         aca.CHECK_DATE,
         ASA.VENDOR_id,
        -- :p_sts,
         aca.amount,
       --  :p_vendor_site,
        --- :p_chk_from_date,
       --  :p_chk_to_date,
       --  :p_cle_from_date,
       --  :p_cle_to_date,
        -- :p_sup_num,
         CASE ASA.VENDOR_NAME
            WHEN 'Petty Cash'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            fnd_global.local_chr (10)),
                              fnd_global.local_chr (10)),
                     fnd_global.local_chr (10))
               || aca.COUNTRY
            WHEN 'Staff Loan'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            fnd_global.local_chr (10)),
                              fnd_global.local_chr (10)),
                     fnd_global.local_chr (10))
               || aca.COUNTRY
            WHEN 'RME Deposit To Other'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            fnd_global.local_chr (10)),
                              fnd_global.local_chr (10)),
                     fnd_global.local_chr (10))
               || aca.COUNTRY
            ELSE
               aca.VENDOR_SITE_CODE
         END
            VENDOR_SITE_CODE,
         --     AIA.INVOICE_NUM,
         (SELECT DECODE (name,
                         'Alrowad Construction_OU', 'Rowad Modern Engineering',
                         name)
            FROM hr_operating_units hou
           WHERE aca.org_id = hou.ORGANIZATION_ID)
            org_name,
         aca.org_id,
         --      AIA.INVOICE_AMOUNT,
         aca.VENDOR_SITE_CODE,
         aca.STATUS_LOOKUP_CODE,
         NVL (aca.FUTURE_PAY_DUE_DATE, aca.CHECK_DATE) maturaty_date,
ai.attribute10 Category
    FROM APPS.AP_CHECKS_ALL ACA,                 --PAYMENT_METHOD,CHECK_NUMBER
         ap_invoices_all ai,
         ap_invoice_payments_all aip,
         --    APPS.AP_INVOICE_PAYMENTS_ALL AIPA,
         APPS.AP_SUPPLIERS ASA,
         ce_bank_accounts cbac,
         ce_bank_acct_uses_all cbau,
         PA_PROJECTS_ALL PAP
   WHERE     aca.VENDOR_ID = ASA.VENDOR_ID
         AND cbau.BANK_ACCT_USE_ID = aca.CE_BANK_ACCT_USE_ID
         AND cbau.BANK_ACCOUNT_ID = cbac.BANK_ACCOUNT_ID
         AND ai.invoice_id(+) = aip.invoice_id
         --and ail.invoice_id = aip.invoice_id
         AND pap.project_id(+) = ai.ATTRIBUTE6
     And pap.PROJECT_STATUS_CODE in 'APPROVED'
         --           AND ail.invoice_id = ai.invoice_id
         AND aip.check_id = aca.check_id
        and cbac.attribute1 != 'NON'
