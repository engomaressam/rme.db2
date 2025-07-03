from docx import Document

doc = Document()
doc.add_heading('AI Explains: Neural Network Scripts and rev.02 Changes', 0)

def add_section(title, original, rev02, impact):
    doc.add_heading(title, level=1)
    doc.add_heading('Original Script:', level=2)
    doc.add_paragraph(original)
    doc.add_heading('rev.02 Changes:', level=2)
    doc.add_paragraph(rev02)
    doc.add_heading('What the Changes Mean:', level=2)
    doc.add_paragraph(impact)

add_section(
    '03a_simple1hiddenlayer',
    """
- A simple neural network with one hidden layer (H=2) is trained to classify 1D data (X in [-20, 20]).
- Uses SGD optimizer, 1000 epochs, and a custom cross-entropy loss.
- The model learns to map a non-linearly separable region to a linearly separable one.
- Plots show the model's output and activations during training.
    """,
    """
- Data range is expanded (X in [-30, 30]), and the hidden layer size is increased (H=4).
- Uses Adam optimizer instead of SGD, and trains for 800 epochs.
- Random seed is changed for different initialization.
    """,
    """
- Increasing the hidden layer size allows the model to learn more complex patterns.
- Adam optimizer adapts the learning rate, often leading to faster or more stable convergence.
- Changing the data range and seed means the model sees a different problem and starts from different weights, so the results and learned boundaries will differ.
    """
)

add_section(
    '03b_multiple_neurons',
    """
- Similar to 03a, but focuses on the effect of more hidden neurons.
- Trains on X in [-20, 20], H=2, SGD, 1000 epochs.
- Shows how increasing neurons can improve the model's ability to fit the data.
    """,
    """
- Data is denser (X in [-10, 10], step 0.5), H=5 (more neurons), Adam optimizer, 600 epochs.
- Random seed changed for new initialization.
    """,
    """
- More neurons = more model capacity, so the network can fit more complex shapes.
- Adam optimizer and denser data can lead to different learning dynamics and potentially better fits.
    """
)

add_section(
    '03c_xor_v2',
    """
- Trains a network to solve the noisy XOR problem with 1, 2, and 3 hidden neurons.
- Data: 100 samples, SGD, 500 epochs, BCELoss.
- Shows how more neurons are needed to solve XOR.
    """,
    """
- Data: 200 samples (more noise), batch size 4, Adam optimizer, 300 epochs.
- Random seed changed.
    """,
    """
- More data and noise make the problem harder, so the model's ability to generalize is tested.
- Adam optimizer and batch training can help with convergence and stability.
- Fewer epochs may mean less overfitting, but also less time to learn.
    """
)

add_section(
    '03d_one_layer_neural_network_MNIST',
    """
- Classifies MNIST digits with a single hidden layer (H=50), SGD, 2 epochs.
- Shows basic image classification with a shallow network.
    """,
    """
- Hidden layer increased to H=128, Adam optimizer, 3 epochs, new seed.
    """,
    """
- More hidden units = more capacity to learn digit features.
- Adam optimizer can speed up and stabilize training.
- More epochs = better accuracy, but risk of overfitting if too many.
    """
)

add_section(
    '03e_activationfuction_v2',
    """
- Visualizes activation functions: Sigmoid, Tanh, ReLU.
- Shows their shapes and how they transform input data.
    """,
    """
- Adds LeakyReLU, expands data range to [-10, 10], new seed.
    """,
    """
- LeakyReLU helps with the "dying ReLU" problem (neurons stuck at zero).
- Wider data range shows more of each function's behavior.
    """
)

add_section(
    '03f_mist1layer_v2',
    """
- Tests different activation functions on MNIST with H=50, SGD, 2 epochs.
- Compares Sigmoid, Tanh, ReLU.
    """,
    """
- H=64, Adam optimizer, 2 epochs, new seed.
- Tries ReLU, Sigmoid, and Tanh in a new order.
    """,
    """
- More hidden units and Adam can improve learning.
- Different activations affect how the network learns features: ReLU is often best for deep nets, Sigmoid/Tanh can saturate.
- Changing order lets you see how each activation impacts accuracy and loss.
    """
)

doc.add_heading('General Notes for Students', level=1)
doc.add_paragraph(
    "- Changing the random seed changes the initial weights, so results may vary.\n"
    "- More hidden units = more model capacity, but also more risk of overfitting.\n"
    "- Adam optimizer is usually faster and more robust than SGD, but can sometimes overfit.\n"
    "- More data or more noise makes the problem harder, but can help generalization.\n"
    "- Try running the scripts yourself and experiment with different values to see how the model behaves!\n"
)

doc.save("AI_explains_neural_net_scripts_and_rev.02.docx")
print("AI explanation file created.") 