import pickle
import json
import numpy as np

with open('scaler.pkl', 'rb') as f:
    s = pickle.load(f)
    data = {
        'mean': s.mean_.tolist(),
        'scale': s.scale_.tolist()
    }
    with open('static/scaler_params.json', 'w') as jf:
        json.dump(data, jf)
print("Scaler parameters saved to static/scaler_params.json")
