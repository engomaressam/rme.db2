import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# MySQL Connection Manager

def connect_to_mysql(host, user, password, database):
    """
    Connect to MySQL database
    
    Args:
        host (str): MySQL host
        user (str): MySQL user
        password (str): MySQL password
        database (str): MySQL database name
        
    Returns:
        mysql.connector.connection.MySQLConnection: MySQL connection
    """
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info("MySQL connection established")
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def create_connection():
    try:
        connection = connect_to_mysql(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DB')
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Example usage
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        conn.close()
