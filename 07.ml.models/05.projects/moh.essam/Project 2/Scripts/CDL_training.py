# training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import traceback
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def file_prompt(msg):
    # Prompt user to select Excel file
    root = Tk()
    root.withdraw()
    ini_file_path = os.getcwd()
    excel_file_path = askopenfilename(title=msg, filetypes=[("Excel Files", "*.xls*")], initialdir=ini_file_path)
    root.destroy()
    return excel_file_path


try:
    # Load the Excel file
    file_path = file_prompt('Please choose your training dataset xlsx file...')
    data = pd.read_excel(file_path)

    # Convert all columns to string type to ensure uniform data type
    data = data.applymap(str)

    # Drop the 'Project' column
    data = data.drop(columns=['Project'])

    # Handle missing values by filling them with a placeholder
    data = data.fillna('Unknown')

    # Get unique values for each feature column
    unique_values = {}
    for column in data.columns:
        unique_values[column] = data[column].unique()

    # Save unique values to a .pkl file
    joblib.dump(unique_values, 'unique_values.pkl')

    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object' and column != 'Main Code':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # Encode the target variable 'Main Code'
    main_code_le = LabelEncoder()
    data['Main Code'] = main_code_le.fit_transform(data['Main Code'])

    # Separate features and target variable
    X = data.drop(columns=['Main Code'])
    y = data['Main Code']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model and the label encoders
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(main_code_le, 'main_code_encoder.pkl')

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Get the unique labels present in y_test and y_pred
    unique_labels = sorted(set(y_test) | set(y_pred))

    # Create a list of class names from main_code_le
    target_names = [str(label) for label in main_code_le.classes_]

    # Generate the classification report
    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
    print(report)
    input('Press enter to exit...')

except Exception as e:
    print(e)
    traceback.print_exc()
    input('Press enter to exit...')
