import pandas as pd

def clean_values(data, official_names):
    # Convert official_names to a set for faster lookup
    official_names_set = set(official_names)

    def find_closest_match(value):
        # Remove brackets and leading/trailing spaces
        cleaned_value = value.strip('[] ') 

        if cleaned_value in official_names_set:
            return cleaned_value

        # If no exact match, find the closest match allowing for max 2 character difference
        for official_name in official_names_set:
            if abs(len(official_name) - len(cleaned_value)) <= 2 and cleaned_value in official_name:
                return official_name

        # If no close match is found, keep the original value
        return value  

    data['Value'] = data['Value'].astype(str).apply(find_closest_match)
    return data

# Read the Excel file into a DataFrame
df_workpackages = pd.read_csv('Active Work Package (Final).xlsx - Active Workpackages.csv')

# Extract the official names from the 'Workpackage' column
official_names = df_workpackages['Workpackage'].tolist()

# Replace 'dataset' with the actual name of your Power BI table
output = clean_values(dataset_from_last_step, official_names)