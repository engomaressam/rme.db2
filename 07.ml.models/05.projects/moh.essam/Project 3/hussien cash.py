from joblib import load
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import datetime as dt
import xlrd


def file_prompt(msg):
    # Prompt user to select Excel file
    root = Tk()
    root.withdraw()
    ini_file_path = os.getcwd()
    excel_file_path = askopenfilename(title=msg, filetypes=[("Excel Files", "*.xls*")], initialdir=ini_file_path)
    root.destroy()
    return excel_file_path


def reordering(df,pos,name):
    column_moved = df[name]
    df.pop(name)
    df.insert(pos, name, column_moved)


def excel_to_datetime(excel_date):
    python_date = xlrd.xldate_as_datetime(excel_date, 0)
    return pd.to_datetime(python_date)

# Load the DataFrame
df_dict = pd.read_excel(file_prompt('Please choose your dictionary file'))
path = file_prompt('Please choose your raw excel file')
if path.split(sep='.')[-1] == 'xlsb':
    df = pd.read_excel(path, engine='pyxlsb')
elif path.split(sep='.')[-1] == 'xlsm':
    df = pd.read_excel(path)
else:
    print('Choose a valid file format')
df = df[16:]
df.columns = df.iloc[0] #fixing header
df = df[1:] #deleting old header
df = df.reset_index(drop=True)


# Load the trained models and encoders
model1 = load('trained_model1.joblib')
label_encoder1 = load('label_encoder1.joblib')

model2 = load('trained_model2.joblib')
label_encoder2 = load('label_encoder2.joblib')

model3 = load('trained_model3.joblib')
label_encoder3 = load('label_encoder3.joblib')

onehot_encoder = load('onehot_encoder.joblib')

# Extract the features from the new data
new_features = df[['   Supplier', 'Category', 'Invoice']]

# Perform one-hot encoding on the new features
encoded_new_features = onehot_encoder.transform(new_features)

# Use the trained model to make predictions for the first column
predicted_labels1 = model1.predict(encoded_new_features)

# Decode the predicted labels for the first column back to their original values
decoded_labels1 = label_encoder1.inverse_transform(predicted_labels1)

# Use the trained model to make predictions for the second column
predicted_labels2 = model2.predict(encoded_new_features)

# Decode the predicted labels for the second column back to their original values
decoded_labels2 = label_encoder2.inverse_transform(predicted_labels2)

# Use the trained model to make predictions for the 3rd column
predicted_labels3 = model3.predict(encoded_new_features)

# Decode the predicted labels for the second column back to their original values
decoded_labels3 = label_encoder3.inverse_transform(predicted_labels3)

df['Log Date'] = None
df['Our Categorization'] = list(decoded_labels1)
df['General Categorization'] = list(decoded_labels2)
df['2nd Categorization'] = list(decoded_labels3)
df['Month'] = df['Check Date'].apply(excel_to_datetime)
df['Month'] = pd.to_datetime(df['Month'])
df['Month_floored'] = df['Month'].dt.to_period('M').dt.to_timestamp()
# columns_keep = ['Project','Doc  Num', 'Predicted Our Categorization','Predicted General Categorization']
# df = df[columns_keep]

# Cleaning dictionary dataframe
df_dict.dropna(axis=0, how='all', inplace=True)
df_dict.reset_index()

#adding 4 more columns to the final output
headers = ['Cost Center', 'Serial', 'Project Name Control', 'Project MANAGER']
#looping through header names that need to be added to df
for header in headers:
    for i, name in enumerate(df['Project']):
        try:
            df.loc[i, header] = df_dict[df_dict['Project Oricale'] == name][header].values[0] # extend used to eliminate list inside lists
        except:
            df.loc[i, header] = 'Unknown ' + header

# Finally reodering
reordering(df, 0, 'Log Date')
reordering(df, 2, 'Cost Center')
reordering(df, 3, 'Serial')
reordering(df, 4, 'Project Name Control')
reordering(df, 5, 'Project MANAGER')
reordering(df, 13, 'Our Categorization')
reordering(df, 14, '2nd Categorization')
reordering(df, 15, 'General Categorization')
reordering(df, 17, 'Month')

df.to_csv('Predicted Results.csv', index=False, encoding='utf-8-sig')