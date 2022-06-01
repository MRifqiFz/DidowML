import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import random
import wget
import zipfile
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import np_utils

# url = "https://storage.googleapis.com/didow-ml-datasets/a-z-handwritten-data.zip"
# url2 = "https://storage.googleapis.com/didow-ml-datasets/mnist-letter.zip"
# filename = wget.download(url)
# filename2 = wget.download(url2)
# local_zip = ['mnist-letter.zip', 'a-z-handwritten-data.zip']

# for i in local_zip:
#     zip_ref = zipfile.ZipFile(i, 'r')

#     zip_ref.extractall('./Dataset/')
#     zip_ref.close()

# Read CSV 
az_data = pd.read_csv('Dataset\A_Z Handwritten Data.csv')
emnist_train = pd.read_csv('Dataset\emnist-letters-train.csv')
emnist_test = pd.read_csv('Dataset\emnist-letters-test.csv')

# Split Data and Label
emnist_train_labels = np.array(emnist_train.iloc[:,0].values) - 1
emnist_train_letters = np.array(emnist_train.iloc[:,1:].values)
emnist_test_labels = np.array(emnist_test.iloc[:,0].values) - 1
emnist_test_letters = np.array(emnist_test.iloc[:,1:].values)
az_data_labels = np.array(az_data.iloc[:,0].values)
az_data_letters = np.array(az_data.iloc[:,1:].values)

def reshape_normalize(data):
    data = data / 255.0
    data = data.reshape(len(data), 28, 28, 1)
    return data

def reshape_rotate(data):
    data = data.reshape(28,28)
    data = np.fliplr(data)
    data = np.rot90(data)
    return data

emnist_train_letters = np.apply_along_axis(reshape_rotate, 1, emnist_train_letters)
emnist_test_letters = np.apply_along_axis(reshape_rotate, 1, emnist_test_letters)
az_data_letters = az_data_letters.reshape(len(az_data_letters), 28, 28)
print('A-Z Handwritten Data: ', az_data_letters.shape)
print('EMNIST Letter Train Data: ', emnist_train_letters.shape)
print('EMNIST Letter Test Data: ', emnist_test_letters.shape)

letters = np.vstack([emnist_train_letters, emnist_test_letters, az_data_letters])
labels = np.hstack([emnist_train_labels, emnist_test_labels, az_data_labels])
print(letters.shape, labels.shape)

x_train, x_test, y_train, y_test = train_test_split(letters, labels, test_size=0.2, random_state=42)

x_train = reshape_normalize(x_train)
x_test = reshape_normalize(x_test)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# def show_image(image, label):
#     image = image.reshape([28, 28])
#     plt.title('Label :' + letters[label])
#     plt.imshow(image)

# n = random.randint(0, len(emnist_train_letters))
# show_image(emnist_train_letters[n], emnist_train_labels[n])

y_train = np_utils.to_categorical(y_train, 26)
y_test = np_utils.to_categorical(y_test, 26)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

opt = tf.keras.optimizers.Adam()

model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose = 2)

print("Test Loss: ",test_loss)
print("Test Accuracy: ",test_accuracy)

model_version = '2'
model_name = 'combine'
file_name = 'combine_model'
file_path = "./Model/{}/{}/{}.h5".format(model_name, model_version, file_name)
model.save(file_path)