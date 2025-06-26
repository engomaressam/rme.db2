import mysql.connector as mysql
 
from mysql.connector import Error
 
 
try:
    # Establish the connection
    cnx = mysql.connect(
            host = "10.10.11.242",
            user ="omaressam",
            password = "Omar@1234$",database = "RME_TEST"
       
    )
   
    if cnx.is_connected():
        print("Connection successful!")
 
except Error as e:
    print(f"Error connecting to database: {e}")
 
finally:
    # Ensure connection is closed properly
    if 'cnx' in locals() and cnx.is_connected():
        cnx.close()
        print("Connection closed.")
   