import os

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"

learning_rate = 0.05

cost_cross = train(Y, X, model, optimizer, criterion_cross, epochs=1500)

# ... existing code ... 