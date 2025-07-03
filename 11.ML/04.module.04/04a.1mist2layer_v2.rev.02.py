import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

hidden_dim1 = 100  # changed from 50
hidden_dim2 = 30   # changed from 50
cust_epochs = 20   # changed from 10 