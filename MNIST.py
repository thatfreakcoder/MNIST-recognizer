# Import Helper Libraries 
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import gradio as gr

# Initialise and Load Dataset
mnist = keras.datasets.mnist
(train_img, train_label), (test_img, test_label) = mnist.load_data()

# Visualise the Test and Train data
print(f'Training Images : {train_img.shape}')
print(f'Testing Images : {test_img.shape}')

# Function for plotting example Image of the Dataset with label
def plot_example(img_set, i):
    plt.figure()
    plt.imshow(img_set[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
plot_example(test_img, 18)

# Preprocess and normalise the Training and Testing Images
train_img, test_img = train_img/255, test_img/255

# Plot first 50 images of the training dataset along with the labels
plt.figure(figsize=(10, 12))
for i in range(50):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i], cmap=plt.cm.binary)
    plt.xlabel(train_label[i])
plt.show()

# Creating a Sequential Model
from tensorflow.keras import Sequential
model = Sequential()

# Adding the Neural Layers
# Flatten -> Dense -> Dense -> Softmax
from tensorflow.keras.layers import Flatten, Dense, Softmax
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Softmax())

# Compiling the Model and view the summary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

# from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.summary()

# Setting the Hyper Parameters
# EPOCHS = 10

# Training the Model
model.fit(train_img, train_label, epochs=10, batch_size=100)

# Evaluate the model against Test Data
test_loss, test_acc = model.evaluate(test_img, test_label, verbose=2)
print(f'Test Accuracy : {str(test_acc*100)[:5]} %')

# Make Prediction on the Model

def classify(image):
    predictions = model.predict(image).tolist()[0]
    return {str(i): predictions[i] for i in range(10)}

sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(classify, sketchpad, label, live=True, capture_session=True)

interface.launch(share=True)


predictions = model.predict(test_img)
for i in range(10, 21):
    predictions[i]
    
    predicted_label = np.argmax(predictions[i])
    
    true_label = test_label[i]

    plot_example(test_img, i)
    print(f'Predicted Value : {predicted_label}')
    print("----------------------------------------")
    
plt.figure(figsize=(15, 20))
for i in range(30, 60):
    plt.subplot(10, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_img[i], cmap=plt.cm.binary)
    plt.xlabel(f'Predicted : {np.argmax(predictions[i])}\nTrue : {test_label[i]}')
plt.show()
    