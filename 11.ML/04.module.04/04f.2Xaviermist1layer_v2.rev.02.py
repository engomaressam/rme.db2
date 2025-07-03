import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

layers = [input_dim, 50, 20, 50, 20, 50, output_dim]  # changed layer sizes
epochs = 25  # changed from 15 