# prediction
import pandas as pd
import joblib
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import traceback
import warnings
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)


def file_prompt(msg):
    # Prompt user to select Excel file
    root = Tk()
    root.withdraw()
    ini_file_path = os.getcwd()
    excel_file_path = askopenfilename(title=msg, filetypes=[("Excel Files", "*.xls*")], initialdir=ini_file_path)
    root.destroy()
    return excel_file_path


def directory_prompt(msg):
    # Prompt user to select a directory
    root = Tk()
    root.withdraw()
    ini_dir_path = os.getcwd()  # Initial directory path
    selected_dir_path = askdirectory(title=msg, initialdir=ini_dir_path)
    root.destroy()
    # Ensure a directory path was selected (filedialog returns an empty string if canceled)
    if selected_dir_path and os.path.isdir(selected_dir_path):
        return selected_dir_path
    else:
        return None  # Return None or handle the case where no valid directory was selected


try:
    pkl_dir = directory_prompt('Please choose the directory that contains your .pkl files')
    cdl_path = file_prompt('Please choose Cost Line Distribution in xlsx format...')

    # fixing cost distribution formatting
    df = pd.read_excel(cdl_path)
    header1 = df.loc[17].tolist()
    header2 = df.loc[18].tolist()

    new_header = []
    for i, h1 in enumerate(header1):
        # get number of nan next to h1
        x = 0
        if i+x+1 < len(header1):
            if str(h1) != 'nan':
                while str(header1[i+x+1]) == 'nan':
                    x+=1
                    if i+x+1 >= len(header1):
                        break
                for j in range(i, i+x+1):
                    new_header.append(str(h1) + "_" + str(header2[j]))

    for i, title in enumerate(new_header):
        if title.split('_')[1] == 'nan':
            new_header[i] = title.split('_')[0]

    df.columns = new_header
    df = df[19:]
    df.reset_index(drop=True, inplace=True)
    df['GL date'] = pd.to_datetime(df['GL date'])

    # Load the saved model, label encoders, and unique values
    rf_model = joblib.load(os.path.join(pkl_dir, 'random_forest_model.pkl'))
    label_encoders = joblib.load(os.path.join(pkl_dir, 'label_encoders.pkl'))
    main_code_le = joblib.load(os.path.join(pkl_dir, 'main_code_encoder.pkl'))
    unique_values = joblib.load(os.path.join(pkl_dir, 'unique_values.pkl'))

    # Load and preprocess the new data
    original_data = df.copy()  # Load the new data correctly

    # Convert all columns to string type to ensure uniform data type
    new_data = original_data.copy()
    new_data = new_data.applymap(str)

    # Handle missing values by filling them with a placeholder
    new_data = new_data.fillna('Unknown')

    # Reorder columns and add 'Project Type'
    new_data = new_data[['Transaction source', 'Task_name', 'Top Task_name', 'Expenditure type', 'Expenditure Category', 'Line Desc', 'Owner']]
    new_data['Project Type'] = 'Residential'
    column_to_move = new_data.pop('Project Type')
    new_data.insert(0, 'Project Type', column_to_move)

    # Preprocess the data using the unique values
    for column in new_data.columns:
        if column in unique_values:
            # Replace new values with '-'
            new_data[column] = new_data[column].apply(lambda x: x if x in unique_values[column] else '-')

    # Encode categorical variables using the saved label encoders
    for column in new_data.columns:
        if column in label_encoders:
            le = label_encoders[column]
            new_data[column] = le.transform(new_data[column])

    # Prepare the feature matrix
    X_new = new_data  # Assuming all columns are features

    # Predict the target variable
    y_pred = rf_model.predict(X_new)

    # Decode the predictions back to original values
    main_code_predictions = main_code_le.inverse_transform(y_pred)

    # Add the decoded predictions to the original data
    original_data['Main Code Predictions'] = main_code_predictions

    # Print or save the DataFrame with predictions
    print(original_data)

    original_data.to_excel('CDL With Predictions.xlsx', index=False)

    print(f'File exported in same directory...')
    input('Press enter to exit...')

except Exception as e:
    print(e)
    input('Press enter to exit...')
    traceback.print_exc()
