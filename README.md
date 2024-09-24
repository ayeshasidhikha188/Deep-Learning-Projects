# 1.  TensorFlow App (Tensorapp.py)
This app allows users to configure a TensorFlow model with various parameters, providing insights into how each one influences the model's performance. 
![image](https://github.com/user-attachments/assets/1e33baf7-887e-49b6-b878-c8b506e3d7e5)



![image](https://github.com/user-attachments/assets/6d532304-4e77-4315-9fd6-8a2a552cb1e0)


Below are the key parameters you can customize:

## Parameters
**1. Learning Rate:** Controls how much to change the model in response to the estimated error during weight updates; affects convergence speed.

**2. Number of Hidden Layers:** Defines the complexity of the model; more layers can capture intricate patterns but may lead to overfitting.

**3. Activation Function:** Determines the output of each neuron in the hidden layers; affects how the model learns non-linear relationships.

**4. Batch Size:** Specifies the number of training samples to process before updating the model parameters; impacts training stability and memory usage.

**5. Regularization Rate:**  Helps prevent overfitting by adding a penalty on the size of coefficients; improves generalization of the model.

## Usage
You can access and use the app through the following link: Hugging Face App Link(https://huggingface.co/spaces/Ayesha188/Tensorflow_Playground)



# 2. Deep Neural Network Explorer
Deep Neural Network Explorer is an intuitive tool designed for exploring and experimenting with various deep neural network configurations tailored for classification tasks. The app enables users to visualize training progress, and evaluate model performance, making it ideal for both educational purposes and model prototyping.
![image](https://github.com/user-attachments/assets/0fff295b-2c23-44a5-9830-a704a91b99eb)


## Key Features and Parameter Descriptions
**Problem  Type** Specially designed for linear and ,non linear datasets , allowing users to configure the learning process to suit their specific dataset.

**Learning Rate:** Controls how much to adjust model weights during training. Fine-tune using options like 1e-05 to optimize convergence.

**Activation Functions:** Choose from multiple activation functions, such as tanh, to introduce non-linearity into the model and help it learn complex patterns.

**Regularization Options:** Helps prevent overfitting by adding penalties to model weights. Adjustable rates (e.g., 1e-05) allow for precise control over the training process.

**Epochs:** Defines the number of complete iterations over the training dataset, allowing customization of training duration (e.g., 50 epochs).

**Test Size:** Specify the percentage split between training and testing data to assess model performance effectively.

**Scalable Hidden Layers:** Configure neural networks with 1 to 10 hidden layers, enhancing the model's capacity to learn from the data.

**Batch Normalization and Dropout:** Optional settings that improve model generalization and robustness. Enable or disable these techniques to reduce overfitting.

**Dropout Rate:** Adjusts the dropout rate between 0.00 and 1.00, randomly dropping neurons during training to prevent overfitting.

**Early Stopping:** Set patience parameters (e.g., 10 epochs) to halt training when no improvement in validation loss is detected, preventing overfitting.

**Weight Initialization:** Use methods like Glorot Normal to set initial weight values, significantly impacting the speed and stability of the model's convergence.

**Neuron Configuration:** Define the number of neurons in each layer (e.g., 5 neurons for Layer 1) to tailor the network’s capacity to learn features.

**Optimizer Selection:** Choose from optimization algorithms like SGD to minimize the loss function and adjust the model's parameters efficiently.

**Batch Size:** Set the number of samples processed before the model’s weights are updated, ranging from 1 to 256. This affects the stability and speed of learning.


## Prerequisites
Python 3.7+
Required libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

## Configure Neural Network Parameters:

* Adjust learning rate, activation functions, and regularization settings.
* Define the number of epochs, test size percentage, and hidden layers.
* Toggle options for batch normalization and dropout, and set the dropout rate.
* Specify early stopping criteria to avoid overfitting.
* Choose your optimizer (e.g., SGD) and weight initialization strategy (e.g., Glorot Normal).
## Train and Evaluate:

* Train the neural network with the configured settings.
* View real-time results, including training loss, accuracy, and other key performance metrics.
## Results and Analysis
* Post-training, the app provides visual and numerical insights into model accuracy, loss trends, and other metrics.
* Comparative graphs help in evaluating different configurations and tuning decisions.

# Explore the App with  Data
Try the app with different datasets on Hugging Face: Explore Deep Neural Network Explorer(https://huggingface.co/spaces/Ayesha188/Optimized_Neural_Network_Framework)
# Watch the App in Action
See how the app works by watching this video: Video Demonstration (https://drive.google.com/file/d/11N9Zt6CJWQDevjBjjvTp_K71yMCkuZoY/view?usp=sharing)
