import os
import tensorflow as tf
from keras.utils import plot_model

model = tf.keras.models.load_model('./Model/emnist/2/emnist_model.h5')
# model_version = '3'
# model_name = 'emnist'
# file_path = "./Model/{}/{}".format(model_name, model_version)
# model.save(file_path)
print(model.summary())