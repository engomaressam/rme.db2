import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load

def file_prompt(msg):
    # Prompt user to select Excel file
    root = Tk()
    root.withdraw()
    ini_file_path = os.getcwd()
    excel_file_path = askopenfilename(title=msg, filetypes=[("CSV files", "*.CSV")], initialdir=ini_file_path)
    root.destroy()
    return excel_file_path

# Load the DataFrame
df = pd.read_csv(file_prompt('Please choose your CSV file that will be used to train your model'))

# Extract the features (input) columns
features = df[['   Supplier', 'Category', 'Invoice']]

# Extract the first label (output) column
labels1 = df['Our Categorization']

# Extract the second label (output) column
labels2 = df['General Categorization']

# Extract the 3rd label (output) column
labels3 = df['2nd Categorization']

# Encode categorical labels into numerical values
label_encoder1 = LabelEncoder()
encoded_labels1 = label_encoder1.fit_transform(labels1)

label_encoder2 = LabelEncoder()
encoded_labels2 = label_encoder2.fit_transform(labels2)

label_encoder3 = LabelEncoder()
encoded_labels3 = label_encoder3.fit_transform(labels3)

# Perform one-hot encoding on the features
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features = onehot_encoder.fit_transform(features)

# Train the first model
model1 = RandomForestClassifier()
model1.fit(encoded_features, encoded_labels1)

# Train the second model
model2 = RandomForestClassifier()
model2.fit(encoded_features, encoded_labels2)

# Train the 3rd model
model3 = RandomForestClassifier()
model3.fit(encoded_features, encoded_labels3)

# Save the models and encoders
dump(model1, 'trained_model1.joblib')
dump(label_encoder1, 'label_encoder1.joblib')
dump(model2, 'trained_model2.joblib')
dump(label_encoder2, 'label_encoder2.joblib')
dump(model3, 'trained_model3.joblib')
dump(label_encoder3, 'label_encoder3.joblib')
dump(onehot_encoder, 'onehot_encoder.joblib')
