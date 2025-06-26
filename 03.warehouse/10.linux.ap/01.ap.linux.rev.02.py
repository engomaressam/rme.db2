import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import mysql.connector as mysql
from mysql.connector import Error
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_data():
    try:
        # Log the start of the data fetching process
        logging.info("Starting data fetch.")

        # Get dates from DateEntry widgets
        start_date = start_date_entry.get_date().strftime('%Y-%m-%d')
        end_date = end_date_entry.get_date().strftime('%Y-%m-%d')

        # Establish the connection with the specified authentication plugin
        cnx = mysql.connect(
            host="10.10.11.242",
            user="omar2",
            password="Omar_54321",
            database="RME_TEST",
        )

        if cnx.is_connected():
            logging.info("Connection successful!")

            # Create a cursor object
            cursor = cnx.cursor()

            # Execute the query
            query = f"""
                SELECT * 
                FROM RME_ap_check_payments_Report 
                WHERE STR_TO_DATE(CHECK_DATE, '%Y-%m-%d') BETWEEN '{start_date}' AND '{end_date}'
            """
            cursor.execute(query)

            # Fetch all rows
            data = cursor.fetchall()

            # Convert to DataFrame for easier Excel export
            df = pd.DataFrame(data, columns=[i[0] for i in cursor.description])

            # Save to Excel
            df.to_excel("output.xlsx", index=False)
            logging.info("Data saved to output.xlsx")

    except Error as e:
        logging.error(f"Error: {e}")

    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cnx.close()
            logging.info("Connection closed.")


# Create main window
root = tk.Tk()
root.title("AP Check Payment Report")

# Start Date
start_date_label = ttk.Label(root, text="Start Date:")
start_date_label.grid(row=0, column=0, padx=5, pady=5)
start_date_entry = DateEntry(root)
start_date_entry.grid(row=0, column=1, padx=5, pady=5)

# End Date
end_date_label = ttk.Label(root, text="End Date:")
end_date_label.grid(row=1, column=0, padx=5, pady=5)
end_date_entry = DateEntry(root)
end_date_entry.grid(row=1, column=1, padx=5, pady=5)

# Fetch Button
fetch_button = ttk.Button(root, text="Export", command=fetch_data)
fetch_button.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()