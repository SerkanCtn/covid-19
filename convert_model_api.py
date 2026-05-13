import tensorflowjs as tfjs
import tensorflow as tf
import os

# Load the model using standard Keras
model = tf.keras.models.load_model('covid_model.keras')

# Convert to TF.js layers format
tfjs.converters.save_keras_model(model, 'static/tfjs_model')

print("Model successfully converted to static/tfjs_model")
