import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

layers = [input_dim, 150, 100, 150, output_dim]  # changed layer sizes
epochs = 40  # changed from 30 