# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, LeakyReLU, PReLU, BatchNormalization
from keras.regularizers import L1, L2, L1L2
from sklearn.datasets import make_classification, make_regression, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import io
import warnings
warnings.filterwarnings("ignore") 

# Title
st.title('Deep Neural Network Explorer')
st.sidebar.title('Deep Neural Network Explorer')

# Problem Type
problem_type = st.sidebar.selectbox('Problem Type', ['Classification', 'Regression', 'Moons', 'Circles'])

# Learning Rate
learning_rate = st.sidebar.selectbox('Learning Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Activation Functions
activation_func = st.sidebar.selectbox(
    'Activation', 
    ['tanh', 'sigmoid', 'linear', 'relu', 'softmax', 'leaky_relu', 'prelu']
)

# Regularization Rate
regularization_rate = st.sidebar.selectbox('Regularization Rate', [0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# Regularization
regularization = st.sidebar.selectbox('Regularization', ['None', 'L1', 'L2', 'Elastic Net'])

# Define Regularizers
if regularization == 'None':
    kernel_regularizer = None
    bias_regularizer = None
elif regularization == 'L1':
    kernel_regularizer = L1(regularization_rate)
    bias_regularizer = L1(regularization_rate)
elif regularization == 'L2':
    kernel_regularizer = L2(regularization_rate)
    bias_regularizer = L2(regularization_rate)
elif regularization == 'Elastic Net':
    kernel_regularizer = L1L2(l1=regularization_rate, l2=regularization_rate)
    bias_regularizer = L1L2(l1=regularization_rate, l2=regularization_rate)

# Epochs
epochs = st.sidebar.number_input("Select number of Epochs", min_value=1, max_value=1000, value=50)

# Split Train/Test
test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=90, value=40, step=1) / 100

# Hidden Layers
hidden_layers = st.sidebar.slider('Number of Hidden Layers', 1, 10, 1)

# Batch Normalization and Dropout Options
apply_bn = st.sidebar.multiselect('Batch Normalization on Layers', [f'Layer {i+1}' for i in range(hidden_layers)])
apply_dropout = st.sidebar.multiselect('Dropout on Layers', [f'Layer {i+1}' for i in range(hidden_layers)])
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.5)

# Early Stopping
early_stopping = st.sidebar.checkbox('Use Early Stopping')
patience = st.sidebar.number_input("Patience for Early Stopping", min_value=1, max_value=50, value=10)

# Weight Initialization
weight_init = st.sidebar.selectbox('Weight Initialization', ['Glorot Normal', 'Glorot Uniform', 'He Normal', 'He Uniform', 'Zeros', 'Constant'])
if weight_init in ['Zeros', 'Constant']:
    st.warning("Using zeros or constant initialization means weights will not update effectively during training, leading to poor performance.")

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(2,)))

# Add hidden layers based on user input
for i in range(hidden_layers):
    neurons = st.sidebar.number_input(f'No of Neurons in Layer {i+1}', min_value=1, max_value=100, value=5)
    layer_activation = activation_func if activation_func in ['tanh', 'sigmoid', 'linear', 'relu', 'softmax'] else None
    
    if activation_func == 'leaky_relu':
        model.add(Dense(units=neurons, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
        model.add(LeakyReLU())
    elif activation_func == 'prelu':
        model.add(Dense(units=neurons, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))
        model.add(PReLU())
    else:
        model.add(Dense(units=neurons, activation=layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer))

    # Apply Batch Normalization
    if f'Layer {i+1}' in apply_bn:
        model.add(BatchNormalization())
    
    # Apply Dropout
    if f'Layer {i+1}' in apply_dropout:
        model.add(Dropout(rate=dropout_rate))

# Final Layer
if problem_type == 'Regression':
    model.add(Dense(units=1, activation='linear'))
else:
    if problem_type in ['Moons', 'Circles']:
        model.add(Dense(units=1, activation='sigmoid'))
    else:
        model.add(Dense(units=1, activation='relu'))

# Select optimizer
optimizer = st.sidebar.selectbox('Optimizer', ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Nadam'])

# Batch Size
batch_size = st.sidebar.slider("Batch Size", 1, 256, 32)

# Dataset Generation and Visualization
if st.sidebar.button('Submit'):
    # Generate dataset based on the problem type
    if problem_type == 'Classification':
        X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, class_sep=2.5, random_state=10)
        st.subheader("Actual Data (Classification)")
    elif problem_type == 'Moons':
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=20)
        st.subheader("Actual Data (Moons)")
    elif problem_type == 'Circles':
        X, y = make_circles(n_samples=1000, noise=0.05, random_state=20)
        st.subheader("Actual Data (Circles)")
    else:
        X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=20)
        st.subheader("Actual Data (Regression)")

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
    st.pyplot(fig)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=20, stratify=y if problem_type != 'Regression' else None)

    # Standardize Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compile the Model
    loss_function = 'mse' if problem_type == 'Regression' else 'binary_crossentropy'
    metrics = ['mse', 'mae'] if problem_type == 'Regression' else ['accuracy']
    model.compile(optimizer=optimizer.lower(), loss=loss_function, metrics=metrics)

    # Model Summary
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    st.text("Model Summary:")
    st.text(buffer.getvalue())
    buffer.close()

    # Display weights and biases for each hidden layer
    for i, layer in enumerate(model.layers[1:-1]):  # Skip input and output layers
        weights, biases = layer.get_weights()
        st.text(f"Layer {i + 1}:")
        st.text(f"Weights: {weights.shape}, Biases: {biases.shape}")

    # Early Stopping Callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience) if early_stopping else None

    # Training the Model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[early_stopping_cb] if early_stopping else None)

    # Plot Loss and Accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    if problem_type != 'Regression':
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Accuracy over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

    # Plot Decision Regions
    if problem_type != 'Regression':
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_decision_regions(X, y, clf=model, legend=2)
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        st.pyplot(fig)
