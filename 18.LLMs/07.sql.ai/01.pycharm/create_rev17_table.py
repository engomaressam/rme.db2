import sqlalchemy
import pandas as pd

engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:PMO%401234@localhost:5432/postgres')
with engine.begin() as conn:  # begin() ensures commit
    conn.execute(sqlalchemy.text('DROP TABLE IF EXISTS public.po_followup_rev17'))
    conn.execute(sqlalchemy.text('CREATE TABLE public.po_followup_rev17 AS SELECT cost_center_number, project_name, po_num, po_status, vendor, approved_date, po_comments, description, uom, unit_price, currency, amount, term, qty_delivered FROM public.po_followup_from_erp'))
    print('Table created!')
    # Check row count
    result = conn.execute(sqlalchemy.text('SELECT COUNT(*) FROM public.po_followup_rev17'))
    count = result.scalar()
    print(f'Row count in po_followup_rev17: {count}')
