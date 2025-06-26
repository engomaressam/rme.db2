create user dwh identified by dwh ; 

grant select on ap_invoices_all  to dwh ;
grant select on ap_invoice_lines  to dwh ;
grant select on ap_invoice_distributions to dwh ;  
grant select on ap_suppliers      to dwh ;
