import joblib
import numpy as np


def predict_single(sample, model_path='models/ensemble_model.joblib', scaler_path='models/scaler.joblib'):
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
sample = np.array(sample).reshape(1, -1)
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
return int(pred[0])
