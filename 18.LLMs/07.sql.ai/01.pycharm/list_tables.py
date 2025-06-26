import sqlalchemy

engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:PMO%401234@localhost:5432/postgres')
with engine.connect() as conn:
    result = conn.execute(sqlalchemy.text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"))
    print([row[0] for row in result])
