#!/usr/bin/env python3
"""
MySQL to PostgreSQL Migration Script
This script copies the 'po_followup_merged' table from MySQL to PostgreSQL.
"""

import mysql.connector
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import time

# MySQL connection configuration
MYSQL_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

# PostgreSQL connection configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'  # Default database
}

# Table name
TABLE_NAME = 'po_followup_merged'

def create_postgres_schema():
    """Create the schema and table in PostgreSQL"""
    postgres_conn = psycopg2.connect(
        host=POSTGRES_CONFIG['host'],
        user=POSTGRES_CONFIG['user'],
        password=POSTGRES_CONFIG['password'],
        database=POSTGRES_CONFIG['database']
    )
    cursor = postgres_conn.cursor()
    
    # Create a new schema for our PO data
    try:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS po_data")
        postgres_conn.commit()
        print("Schema 'po_data' created or already exists.")
    except Exception as e:
        print(f"Error creating schema: {e}")
        postgres_conn.rollback()
        return False
    
    try:
        # Get table structure from MySQL
        mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        mysql_cursor = mysql_conn.cursor(dictionary=True)
        
        # Get column information
        mysql_cursor.execute(f"DESCRIBE {TABLE_NAME}")
        columns = mysql_cursor.fetchall()
        
        # Map MySQL data types to PostgreSQL
        type_mapping = {
            'int': 'integer',
            'varchar': 'varchar',
            'text': 'text',
            'date': 'date',
            'decimal': 'numeric',
            'double': 'double precision',
            'timestamp': 'timestamp',
            'datetime': 'timestamp',
            'tinyint': 'smallint',
            'smallint': 'smallint',
            'mediumint': 'integer',
            'bigint': 'bigint',
            'float': 'real',
            'char': 'char'
        }
        
        # Create table in PostgreSQL
        create_table_query = f"CREATE TABLE IF NOT EXISTS po_data.{TABLE_NAME} ("
        
        for i, column in enumerate(columns):
            # Extract MySQL data type and convert it to PostgreSQL
            mysql_type = column['Type']
            field_type = None
            
            # Handle varchar with length
            if 'varchar' in mysql_type:
                length = mysql_type.split('(')[1].split(')')[0]
                field_type = f"varchar({length})"
            # Handle decimal with precision
            elif 'decimal' in mysql_type:
                precision = mysql_type.split('(')[1].split(')')[0]
                field_type = f"numeric({precision})"
            # Map other data types
            else:
                for mysql_t, pg_t in type_mapping.items():
                    if mysql_t in mysql_type:
                        field_type = pg_t
                        break
                        
            if field_type is None:
                field_type = 'text'  # Default to text if no match
                
            # Add column to create table query
            create_table_query += f"\"{column['Field']}\" {field_type}"
            
            # Handle NULL constraint
            if column['Null'] == 'NO':
                create_table_query += " NOT NULL"
                
            # Handle primary key
            if column['Key'] == 'PRI':
                create_table_query += " PRIMARY KEY"
                
            # Add comma if not the last column
            if i < len(columns) - 1:
                create_table_query += ", "
        
        create_table_query += ")"
        
        # Execute create table query
        cursor.execute(create_table_query)
        postgres_conn.commit()
        print(f"Table 'po_data.{TABLE_NAME}' created successfully.")
        
        mysql_cursor.close()
        mysql_conn.close()
        
    except Exception as e:
        print(f"Error creating table: {e}")
        postgres_conn.rollback()
        return False
    finally:
        cursor.close()
        postgres_conn.close()
        
    return True

def migrate_data():
    """Copy data from MySQL to PostgreSQL using pandas as intermediary"""
    start_time = time.time()
    
    try:
        # Create SQLAlchemy connection strings
        mysql_uri = f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
        # Properly escape special characters in password
        postgres_uri = f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password'].replace('@', '%40')}@{POSTGRES_CONFIG['host']}/{POSTGRES_CONFIG['database']}"
        
        # Setup connections
        mysql_engine = create_engine(mysql_uri)
        postgres_engine = create_engine(postgres_uri)
        
        # Read data from MySQL
        print(f"Reading data from MySQL table '{TABLE_NAME}'...")
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", mysql_engine)
        
        row_count = len(df)
        print(f"Read {row_count} rows from MySQL table.")
        
        # Write data to PostgreSQL
        print(f"Writing data to PostgreSQL table 'po_data.{TABLE_NAME}'...")
        df.to_sql(
            name=TABLE_NAME, 
            schema='po_data',
            con=postgres_engine, 
            if_exists='replace',  # Replace if table already exists
            index=False
        )
        
        # Verify row count
        try:
            with psycopg2.connect(
                host=POSTGRES_CONFIG['host'],
                user=POSTGRES_CONFIG['user'],
                password=POSTGRES_CONFIG['password'],
                database=POSTGRES_CONFIG['database']
            ) as conn:
                with conn.cursor() as curs:
                    curs.execute(f"SELECT COUNT(*) FROM po_data.{TABLE_NAME}")
                    pg_row_count = curs.fetchone()[0]
                    
                    if pg_row_count == row_count:
                        print(f"[SUCCESS] Data migration successful! {row_count} rows copied.")
                    else:
                        print(f"[WARNING] Row count mismatch: MySQL has {row_count} rows, PostgreSQL has {pg_row_count} rows.")
        except Exception as e:
            print(f"[WARNING] Could not verify row count: {e}")
            print(f"Data migration completed, but count verification failed.")
            # Assume success since data was written
            
        end_time = time.time()
        print(f"Migration completed in {end_time - start_time:.2f} seconds.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during data migration: {e}")
        return False

def main():
    print("Starting MySQL to PostgreSQL migration...")
    
    if create_postgres_schema():
        migrate_data()
    else:
        print("Failed to create schema/table. Migration aborted.")
    
    print("Migration process complete.")
    
if __name__ == "__main__":
    main()
