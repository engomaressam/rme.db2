#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import pandas as pd
import tkinter as tk
from tkinter import simpledialog


# In[2]:


# Database connection parameters
db_params = {
    "host": "localhost",
    "dbname": "rme",
    "user": "postgres",
    "password": "omar_321"
}

# Connect to the PostgreSQL server
connection = psycopg2.connect(**db_params)
cursor = connection.cursor()


# In[3]:


# Create a Tkinter root window (hidden)
root = tk.Tk()
root.withdraw()

# Prompt the user to enter the project number using a dialog box
project_no_value = simpledialog.askstring("Project Number", "Please enter the project number:")


# In[4]:


# SQL query to select data from the table where project_no matches user input
query = "SELECT * FROM cost_dist WHERE project_no = %s"
cursor.execute(query, (project_no_value,))

# Fetch all rows
rows = cursor.fetchall()

# Create a pandas DataFrame from the fetched data
df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])


# In[5]:


# Export the data to an Excel file
excel_filename = f"cost_dist_{project_no_value}.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Data exported to '{excel_filename}' successfully!")

# Close the database connection
cursor.close()
connection.close()
print("Database connection closed.")

