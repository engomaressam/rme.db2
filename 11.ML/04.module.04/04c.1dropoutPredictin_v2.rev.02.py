import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

model_drop = Net(2, 300, 2, p=0.7)  # changed dropout from 0.5 to 0.7
epochs = 700  # changed from 500 