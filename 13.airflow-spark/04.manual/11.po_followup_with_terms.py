import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, String, Date, DECIMAL, Text

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

def main():
    # Create SQLAlchemy engine
    engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}")

    # Read tables into DataFrames
    po_followup_df = pd.read_sql("SELECT * FROM po_followup_for_chatbot", engine)
    po_terms_df = pd.read_sql("SELECT PO_NUMBER, LONG_TEXT FROM po_terms", engine)

    # Merge on po_number (left join)
    merged_df = pd.merge(
        po_followup_df,
        po_terms_df,
        how='left',
        left_on='po_number',
        right_on='PO_NUMBER'
    )
    if 'PO_NUMBER' in merged_df.columns:
        merged_df.drop(columns=['PO_NUMBER'], inplace=True)

    # Select only the columns you want
    expected_cols = ['id', 'po_number', 'creation_date', 'line_amount', 'vendor_name', 'project_name', 'LONG_TEXT']
    merged_df = merged_df[expected_cols]

    # Diagnostics (optional, can be commented out)
    print("Sample row:", merged_df.iloc[0].to_dict())
    print("DataFrame dtypes:\n", merged_df.dtypes)
    print("Number of rows:", len(merged_df))

    # Full insert
    merged_df.to_sql('po_followup_with_terms', con=engine, if_exists='replace', index=False, dtype={
        'id': Integer(),
        'po_number': String(50),
        'creation_date': Date(),
        'line_amount': DECIMAL(18, 2),
        'vendor_name': String(255),
        'project_name': String(255),
        'LONG_TEXT': Text()
    })
    print("Full insert succeeded.")

    # Print stats
    with_terms = merged_df['LONG_TEXT'].notnull().sum()
    without_terms = merged_df['LONG_TEXT'].isnull().sum()
    print(f"POs with terms: {with_terms}")
    print(f"POs without terms: {without_terms}")

if __name__ == "__main__":
    main() 