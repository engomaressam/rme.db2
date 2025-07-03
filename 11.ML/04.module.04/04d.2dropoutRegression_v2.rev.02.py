import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

model = Net(1, 500, 1)
model_drop = Net(1, 500, 1, p=0.5)
epochs = 800  # changed from 500 