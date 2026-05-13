import tensorflow as tf
import json
import numpy as np

# Load the model
model = tf.keras.models.load_model('covid_model.keras')

# Get weights from all layers
weights_data = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        w, b = layer.get_weights()
        weights_data[f'layer_{i}_weights'] = w.tolist()
        weights_data[f'layer_{i}_bias'] = b.tolist()

with open('static/model_weights.json', 'w') as f:
    json.dump(weights_data, f)

print("Model weights exported to static/model_weights.json")
