create or replace  view RME_Collection_grp_by_prj  as ( Select cash_receipt_id,
       receipt_number,
       receipt_date,
       TYPE,
       amount,
       func_amount,
       unidentified_amount,
       applied_amount,
       on_account_amount,
       unapplied_amount,
       (unidentified_amount * conv_rate) unidentified_func_amount,
       (applied_amount * conv_rate) applied_func_amount,
       (on_account_amount * conv_rate) on_account_func_amount,
       (unapplied_amount * conv_rate) unapplied_func_amount,
       currency_code,
       STATUS,
       customer_id,
       activity_id,
       comments,
       rec_no,
       old_no,
       remit_bank_acct_use_id,
       receipt_method_id,
       payment_method,
       project,
       owner
  From (Select Distinct cr.cash_receipt_id,
                        cr.receipt_number,
                        cr.receipt_date,
                        DECODE(cr.TYPE, 'CASH', 'Standard', 'Miscellaneous') TYPE,
                        cr.amount,
                        (cr.amount * NVL(cr.exchange_rate, 1)) func_amount,
                        (Select SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED)) --nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
                           From ar_receivable_applications_all app,
                                AR_PAYMENT_SCHEDULES_ALL       PS_INV
                          Where APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
                            And app.status = 'UNID'
                            And app.cash_receipt_id = cr.cash_receipt_id
                           -- start by mohamed.dagher 12-02-2015 to include show amount by history
                           ---shark     
                           --  And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
                           --  And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date)   Or :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
                           --- --SHARK 
                         ) unidentified_amount,
                        (Select SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED)) --nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
                           From ar_receivable_applications_all app,
                                AR_PAYMENT_SCHEDULES_ALL       PS_INV -- start by mohamed.dagher 31-05-2016 to filter by bu
                         --Started By A.Zaki 29-01-2018
                         --,RA_CUST_TRX_TYPES_all CTI
                         --Ended By A.Zaki 29-01-2018
                         -- end by mohamed.dagher
                          Where APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
                            And app.status In ('APP', 'ACTIVITY')
                            And app.cash_receipt_id = cr.cash_receipt_id
                               -- start by mohamed.dagher 12-02-2015 to include show amount by history
                           --  And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
                           --  And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date)   Or :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
                         -- start by mohamed.dagher 12-02-2015 to include show amount by history
                         --Started By A.Zaki 29-01-2018
                         --And CTI.CUST_TRX_TYPE_ID(+) = PS_INV.CUST_TRX_TYPE_ID
                         --And (ctI.attribute2 = :p_bu Or :p_bu Is Null)-- end by mohamed.dagher 12-02-2015 to include show amount by history
                         --Ended By A.Zaki 29-01-2018
                         ) applied_amount,
                        (Select SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED)) --nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
                           From ar_receivable_applications_all app,
                                AR_PAYMENT_SCHEDULES_ALL       PS_INV
                          Where APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
                            And app.status = 'ACC'
                            And app.cash_receipt_id = cr.cash_receipt_id
                               -- start by mohamed.dagher 12-02-2015 to include show amount by history
                            ---shark  
                         --   And (TRUNC(app.gl_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
                         --   And (TRUNC(app.gl_DATE) <= TRUNC(:p_to_date) Or   :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
                            ---shark  
                         ) on_account_amount,
                        (Select SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED)) --nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
                           From ar_receivable_applications_all app,
                                AR_PAYMENT_SCHEDULES_ALL       PS_INV
                          Where APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
                            And app.status = 'UNAPP'
                            And app.cash_receipt_id = cr.cash_receipt_id
                               -- start by mohamed.dagher 12-02-2015 to include show amount by history
                            -----shark     
                            -- And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
                            -- And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date) Or   :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
                         ) unapplied_amount,
                        NVL(cr.exchange_rate, 1) conv_rate,
                        cr.currency_code,
                        --arpt_sql_func_util.get_lookup_meaning('RECEIPT_CREATION_STATUS', crh_current.status) status,
                        
                        (Select OTR.STATUS
                           From AR_CASH_RECEIPT_HISTORY_ALL OTR
                          Where OTR.CASH_RECEIPT_HISTORY_ID =
                                (Select MAX(INR.CASH_RECEIPT_HISTORY_ID)
                                   From AR_CASH_RECEIPT_HISTORY_ALL INR
                                  Where INR.CASH_RECEIPT_ID =
                                        CR.CASH_RECEIPT_ID
                                    --- SHARK 
                                    --And (INR.GL_DATE >= :P_FROM_DATE Or    :P_FROM_DATE Is Null)
                                    --And (INR.GL_DATE <= :P_TO_DATE   Or    :P_TO_DATE Is Null)
                                    )
                                    ) STATUS,
                        party.party_id customer_id,
                        cr.receivables_trx_id activity_id,
                        cr.comments,
                        cr.attribute1 rec_no,
                        cr.attribute2 old_no,
                        cr.remit_bank_acct_use_id,
                        cr.receipt_method_id,
                        rec_method.name payment_method,
                        pap.name project,
                        nvl((select distinct full_name
                              from per_all_people_f   paf,
                                   pa_project_players ppp
                             where paf.person_id = ppp.person_id
                               and pap.project_id = ppp.project_id
                               AND SYSDATE BETWEEN paf.EFFECTIVE_START_DATE AND
                                   paf.EFFECTIVE_END_DATE
                               and rownum = 1
                               and PROJECT_ROLE_TYPE = '1000'),
                            'General') owner
          From hz_cust_accounts cust,
               hz_parties       party,
               
               ar_receipt_methods   rec_method,
               ar_cash_receipts_all cr,
               PA_PROJECTS_ALL      PAP,
               
               ar_cash_receipt_history_all    crh_current -- START BY MOHAMED.DAGHER 31-05-2016 TO FILTER BY BUSINESS UNTI
              ,
               AR_RECEIVABLE_APPLICATIONS_all APP,
               AR_PAYMENT_SCHEDULES_all       PS_INV,
               RA_CUST_TRX_TYPES_all          CTT
        -- END BY MOHAMED.DAGHER              --,
        /*
        (select cash_receipt_id, status, sum(amount_applied) app_amount
         from   ar_receivable_applications_all app
         group by cash_receipt_id, status) receipt_apply
                   */
         Where cr.pay_from_customer = cust.cust_account_id(+)
           And cust.party_id = party.party_id(+)
           And crh_current.cash_receipt_id = cr.cash_receipt_id
           And cr.receipt_method_id = rec_method.receipt_method_id
           --AND    receipt_apply.cash_receipt_id = cr.cash_receipt_id
           AND cr.attribute1 = pap.segment1(+)
           And crh_current.current_record_flag = 'Y'
             And UPPER(arpt_sql_func_util.get_lookup_meaning('RECEIPT_CREATION_STATUS',crh_current.status)) != 'REVERSED'
          --- And (DECODE(cr.TYPE, 'CASH', 'STANDARD', 'MISCELLANEOUS') =  :p_receipt_type Or :p_receipt_type Is Null)
          --- And (TRUNC(cr.receipt_date) >= TRUNC(:p_from_date) Or  :p_from_date Is Null)
          --- And (TRUNC(cr.receipt_date) <= TRUNC(:p_to_date)   Or  :p_to_date Is Null)
          --- And (cust.cust_account_id = :p_customer Or :p_customer Is Null)
          --- And (cr.receivables_trx_id = :p_activity_id Or :p_activity_id Is Null)
              --AND (:p_status = UPPER(arpt_sql_func_util.get_lookup_meaning('RECEIPT_CREATION_STATUS', crh_current.status)) or :p_status is null)
         
         
              -- START BY MOHAMED.DAGHER 31-05-2016 TO FILTER BY BUSINESS UNTI
           And APP.CASH_RECEIPT_ID = CR.CASH_RECEIPT_ID
           And APP.APPLIED_PAYMENT_SCHEDULE_ID =  PS_INV.PAYMENT_SCHEDULE_ID(+)
              --     And APP.DISPLAY = 'Y'
           And CTT.CUST_TRX_TYPE_ID(+) = PS_INV.CUST_TRX_TYPE_ID
           And APP.CASH_RECEIPT_ID not in
               (SELECT xlt.source_id_int_1
                  FROM apps.xla_events xe, xla.xla_transaction_entities xlt
                 WHERE xe.entity_id = xlt.entity_id
                   AND xe.application_id = xlt.application_id
                   AND xlt.source_application_id = 222
                   AND xlt.ledger_id = 2030
                   AND xe.process_status_code <> 'P'
                   AND xe.event_status_code <> 'P'))
                   
                   )


