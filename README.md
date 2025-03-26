# Handwritten Digit Recognition using Neural Network

# Overview

This project implements a Handwritten Digit Recognition model using a Neural Network. The model is trained on the MNIST dataset **(provided as a .mat file)** to classify handwritten digits (0-9). It achieves high accuracy using a fully connected neural network (MLP) and is implemented using TensorFlow/Keras.

# Dataset

[click here for Dataset](https://www.kaggle.com/datasets/subho117/handwritten-digit-recognition-using-neural-network?select=mnist-original.mat)

The dataset used in this project is the MNIST handwritten digits dataset, stored in a **.mat** file format. It consists of:

- 70,000 grayscale images of handwritten digits

- Each image is 28x28 pixels, flattened into a 784-dimensional vector

- 10 classes (digits 0-9)

- Labels stored as a single array of size (70,000,)

The dataset is preprocessed by normalizing pixel values (0 to 1) and converting labels into one-hot encoding for training the model.

# Installation & Setup

To run this project in Google Colab or Jupyter Notebook, install the required dependencies:
```
pip install numpy scipy tensorflow scikit-learn matplotlib
```

# Model Architecture

The Neural Network consists of:

- **Flatten layer**: Converts 28×28 images into a 1D vector

- Dense (128 neurons, ReLU activation)

- Dense (64 neurons, ReLU activation)

- Dense (10 neurons, Softmax activation) (output layer for classification)

# Training & Evaluation

- The dataset is split (80-20) into training and test sets.

- Categorical cross-entropy is used as the loss function.

- Adam optimizer is used for training.

- The model is trained for 10 epochs.

- Achieves high test accuracy (~98%).

# Visualization

Training accuracy and loss are plotted for better understanding:
```
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

# Testing the Model

You can test the model on new handwritten digits from the dataset:
```
def predict_sample(index):
    sample = X_test[index].reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(sample))
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {prediction}")
    plt.show()

predict_sample(197)
```

# Saving and Loading the Model
```
model.save("handwritten_digit_model.h5")  # Save the trained model
model = tf.keras.models.load_model("handwritten_digit_model.h5")  # Load the saved model
```

# Future Improvements

- Implement a Convolutional Neural Network (CNN) for improved accuracy.

- Develop a GUI using Tkinter for interactive digit recognition.

- Train on additional handwritten datasets for better generalization.
