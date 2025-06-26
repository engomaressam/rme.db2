SELECT DISTINCT
         ACR.cash_receipt_id receipt_id,
         ACR.receipt_number,
         trx_head.org_id,
         ppa.name receipt_prj_name,
         acr.attribute1 receipt_prj_code,
         ACR.amount receipt_amount,
         acr.RECEIPT_DATE,
         trx_head.CUSTOMER_TRX_ID,
         TRX_HEAD.TRX_NUMBER Inv_num,
         ARA.AMOUNT_APPLIED,
         ARA.attribute1,
           ( select nvl(sum(AMOUNT),0)
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ('With Holding Tax')) With_Holding_Tax,

  ( select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ( 'Stamps','Stamp Tax')) Stamp,

    (select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ('Retention AR.3rd Party', 'Retention AR.Affiliates', 'Retention  Affiliates', 'Perform-Reten-3rd Party',
                   'Retention - 3th Party', 'Retention.Final - 3th Party', 'AR Retention 3rd Party', 'Retention 3rd Party', 'TOAC')) Retentionn,

      (select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ('Social insurance','Social Insurans','Social insurance')) Social,

      (select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ('Add Tax','Sales Tax','Tax Adjustment','Taxes')) TAX,

      (select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME not in ('Retention AR.3rd Party', 'Retention AR.Affiliates', 'Retention  Affiliates', 'Perform-Reten-3rd Party',
                        'Stamps', 'Stamp Tax', 'Add Tax', 'Sales Tax', 'With Holding Tax', 'Retention - 3th Party', 'Retention.Final - 3th Party',
                        'AR Retention 3rd Party', 'Retention 3rd Party', 'TOAC', 'Social insurance', 'Social Insurans', 'Social insurance', 
                        'Tax Adjustment', 'Taxes')) OTHER_CONDITIONS,

         TRX_HEAD.INVOICE_CURRENCY_CODE CURRENCY,
         (SELECT SUM (
                      NVL (tl.QUANTITY_INVOICED, 0) * NVL (tl.UNIT_SELLING_PRICE, 1))
          FROM RA_CUSTOMER_TRX_LINES_ALL TL
          WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID) Transaction_Amount,

         (SELECT NVL (SUM (extended_amount), 0)
          FROM ra_customer_trx_all trx_head, ra_customer_trx_lines_all trx_line
          WHERE trx_head.customer_trx_id = trx_line.customer_trx_id(+)
          AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
          AND trx_line.line_type IN ('TAX')) Tax_Amount,

         TO_CHAR (TRX_HEAD.TRX_DATE, 'DD-MM-YYYY') TRX_DATE,
         ara.apply_date,
         TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS') DUE_DATE_DFF,

         hcaa.ACCOUNT_NUMBER Customer_No,
         hcaa.ACCOUNT_NAME Customer_Name
---------------------------------------------------------------------------------
FROM AR_CASH_RECEIPTS_ALL acr,
     pa_projects_all ppa,
     AR_RECEIVABLE_APPLICATIONS_ALL ara,
     RA_CUSTOMER_TRX_ALL trx_head,
     RA_CUSTOMER_TRX_LINES_ALL trx_line,
     HZ_CUST_ACCOUNTS hcaa,
     hr_operating_units hro
-------------------------------------------------------------------------------------
WHERE acr.cash_receipt_id = ara.cash_receipt_id(+)
      AND ppa.segment1 = acr.attribute1
      AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
      AND HCAA.CUST_ACCOUNT_ID(+) = trx_head.BILL_TO_CUSTOMER_ID
      AND ara.reversal_gl_date IS NULL
      AND ara.status IN ('ACC', 'APP')
      AND trx_head.org_id = trx_line.org_id(+)
      AND acr.org_id = hro.organization_id 
      AND hro.organization_id = 83
--------------------------------------------------------------------------------------
GROUP BY ACR.cash_receipt_id,
         ACR.receipt_number,
         trx_head.org_id,
         ppa.name,
         acr.attribute1,
         ACR.amount,
         acr.RECEIPT_DATE,
         ara.apply_date,
         trx_head.CUSTOMER_TRX_ID,
         ara.APPLIED_CUSTOMER_TRX_ID,
         ara.cash_receipt_id,
         TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1,
         ARA.AMOUNT_APPLIED,
         ara.attribute1,
         ara.status,
         TRX_HEAD.EXCHANGE_RATE,
         TRX_HEAD.CUST_TRX_TYPE_ID,
         TRX_HEAD.TRX_NUMBER,
         TRX_HEAD.TRX_DATE,
         TRX_HEAD.INVOICE_CURRENCY_CODE,
         TRX_HEAD.ATTRIBUTE2,
         hcaa.ACCOUNT_NUMBER,
         hcaa.ACCOUNT_NAME,
         ara.APPLIED_PAYMENT_SCHEDULE_ID,
         TRX_HEAD.TERM_ID;
