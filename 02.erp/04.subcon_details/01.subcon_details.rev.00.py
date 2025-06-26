import pandas as pd
import os
from tkinter import Tk, messagebox, ttk
from tkinter.filedialog import askopenfilename
import time
import datetime

def reordering(dataframe, pos, name):
    column_moved = dataframe[name]
    dataframe.pop(name)
    dataframe.insert(pos, name, column_moved)

def file_prompt(msg):
    # Prompt user to select Excel file
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    ini_file_path = os.getcwd()
    excel_file_path = askopenfilename(title=msg, filetypes=[("Excel Files", "*.xls*")], initialdir=ini_file_path)
    root.destroy()
    return excel_file_path

try:
    start_time = time.time()
    root = Tk()
    root.title("Subcontractor_Details.exe (RME)")
    label1 = ttk.Label(root, text='Status: Opening Files')
    label1.grid(row=0, column=0, columnspan=2, sticky='w')
    progress_bar = ttk.Progressbar(root, orient='horizontal', length=450, mode='indeterminate')
    progress_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=15)
    label2 = ttk.Label(root, text='0 Seconds: ')
    label2.grid(row=2, column=0, sticky='w')
    label3 = ttk.Label(root, text='Reading files')
    label3.grid(row=3, column=0, sticky='w')
    label4 = ttk.Label(root, text='')
    label4.grid(row=4, column=0, sticky='w')
    label5 = ttk.Label(root, text='')
    label5.grid(row=5, column=0, sticky='w')
    root.attributes('-topmost', True)
    progress_bar.start()
    root.update()

    df_dict = pd.read_excel(file_prompt('Please choose your dictionary file'))
    df = pd.read_excel(file_prompt('Please choose your raw contract_details file'))

    progress_bar.start()
    root.update()

    df = df[15:]
    df.columns = df.iloc[11]
    # df.rename(columns={'Description': 'Des'}, inplace=True)
    df.columns.values[1] = 'Des'
    df.reset_index(drop=True)
    df['sc_no'] = None
    df['sc_manual'] = None
    df['proj_name'] = None
    df['cost_center'] = None
    df['Project Name Control'] = None
    df['Project MANAGER'] = None
    df['supplier_name'] = None
    df['currency'] = None
    df['supplier_no'] = None
    df['work_package'] = None

    progress_bar.destroy()
    progress_bar = ttk.Progressbar(root, orient='horizontal', length=450, mode='determinate')
    progress_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=15)
    label1.config(text='Running')
    root.update()

    pos_sc = 0
    total_rows = len(df)
    for i, row in df.iterrows():
        if row[0] == 'Line':
            pos_sc = i-11
        if isinstance(row[0], int):
            row['sc_no'] = df.loc[pos_sc, 'Des']
            row['sc_manual'] = df.loc[pos_sc, 'Pay Item No']
            row['proj_name'] = df.loc[pos_sc+1, 'Des']
            row['cost_center'] = df.loc[pos_sc+1, 'Des'].split()[-1].split('-')[-1]
            row['supplier_name'] = df.loc[pos_sc+1, 'Pay Item No']
            row['currency'] = df.loc[pos_sc+2, 'Des']
            row['supplier_no'] = df.loc[pos_sc+2, 'Pay Item No']
            row['work_package'] = df.loc[pos_sc+2, 'Des']

        progress_bar['value'] = (i/total_rows)*100
        elapsed_time = time.time() - start_time
        label2.config(text=f"Elapsed time: {elapsed_time:.2f} seconds")
        label3.config(text=f"Looping through rows {i}")
        label4.config(text=f"Project name: {row['proj_name']}")
        label5.config(text=f"Supplier name: {row['supplier_name']}")
        root.update_idletasks()

    df = df[(df['sc_no'].notna()) & (df['sc_no'] != 'Line')]

    #adding cost center from dictionary
    label1.config(text='Status: Adding cost center and projects manager details')
    label5.destroy()
    root.update()
    headers = ['Project Name Control', 'Project MANAGER']
    c = 0
    total_rows = len(headers) * len(df['sc_no'])
    #looping through header names that need to be added to df
    for header in headers:
        for i, row in df.iterrows():
            try:
                cs = int(row['cost_center'])
                row[header] = df_dict[df_dict['Cost Center'] == cs][header].values[0]
            except:
                row[header] = 'Unknown ' + header
            c += 1
            progress_bar['value'] = (c/total_rows)*100
            label3.config(text=f"Looping through rows {c}")
            label4.config(text=f"Project Name: {row['proj_name']}")
            root.update()

    reordering(df, 1, 'sc_no')
    reordering(df, 2, 'sc_manual')
    reordering(df, 3, 'proj_name')
    reordering(df, 4, 'cost_center')
    reordering(df, 5, 'Project MANAGER')
    reordering(df, 6, 'Project Name Control')
    reordering(df, 7, 'supplier_name')
    reordering(df, 8, 'currency')
    reordering(df, 9, 'supplier_no')
    reordering(df, 10, 'work_package')
    df.to_excel(f'Subcontractor_Details_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx')
    messagebox.showinfo("Success", "EXE Ran Successfully")
except Exception as e:
    messagebox.showerror("Error", f"An error occurred: {e}")

root.destroy()
