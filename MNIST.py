# Import Helper Libraries
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

# Initialise and Load Dataset
mnist = keras.datasets.mnist
(train_img, train_label), (test_img, test_label) = mnist.load_data()
 
# Plot one of the Image of the Dataset with label
plt.figure()
plt.imshow(train_img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print(train_label[0])
    
# Preprocess and normalise the Training and Testing Images
train_img = train_img / 255
test_img = test_img / 255

# Plot first 25 images of the training dataset along with the labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary)
    plt.xlabel(train_label[i])
plt.show()

# Creating the Neural Network Layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])

# Compiling the ML Model
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Making Predictions and Evaluating Loss and Accuracy
model.fit(train_img, train_label, epochs=10)

test_loss, test_acc = model.evaluate(test_img, test_label, verbose=2)
print(f'Test Accuracy : {test_acc}')

# New Model for Making Predictions on Unseen data
new_model = keras.models.Sequential([model,
                                     keras.layers.Softmax()]) # Softmax convert the logits to probabilities, which are easier to interpret.

# Make Prediction on the new Model
predictions = new_model.predict(test_img)
for i in range(10, 21):
    predictions[i]
    
    predicted_label = np.argmax(predictions[i])
    
    true_label = test_label[i]
    
    print(f'Predicted Value : {predicted_label}')
    print(f'Actual Value : {true_label}')
    print("-------------------------------")
    
# Helper Functions for Data Visualization
def plot_img(i, prediction_array, true_label, img):
    true_label, img = true_label[i], img[i]
    
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(prediction_array)
    
    if predicted_label == true_label:
        color = 'blue'

    elif predicted_label != true_label:
        color = 'red'
    prediction_result = str(100 * np.max(prediction_array))
    plt.xlabel(f'Predicted : {predicted_label} - {prediction_result[:5]}% \n True Value : {true_label}', color=color)
    
def plot_value(i, prediction_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
# Predict and Visualize the Result for First 25 
rows = 10
cols = 4
images_num = rows * cols

plt.figure(figsize=(2*2*cols, 2*rows))

for i in range(images_num):
    plt.subplot(rows, 2*cols, 2*i+1)
    plot_img(i, predictions[i], test_label, test_img)
    plt.subplot(rows, cols*2, 2*i+2)
    plot_value(i, predictions[i], test_label)
    
plt.tight_layout()
plt.show()
