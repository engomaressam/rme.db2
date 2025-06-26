select  acr.CASH_RECEIPT_ID,
        acr.RECEIPT_DATE,
        acr.attribute1   Project_code ,
        pap.NAME as Project_Name,
        acr.amount       Recipt_amount_currency ,
        acr.currency_code  Currency ,
        nvl(acr.exchange_rate ,1) exchange_rate   ,
        nvl(acr.amount,0)  *  nvl(acr.exchange_rate ,1) Recipt_amount_EGP , 
        acr.comments
from    AR_CASH_RECEIPTS_all acr
left join apps.pa_projects_all pap ON acr.attribute1 = pap.SEGMENT1
where   acr.ATTRIBUTE_CATEGORY = '83'
and     acr.ATTRIBUTE1 is not null
and     acr.ATTRIBUTE2 != 'Manual/Netting'