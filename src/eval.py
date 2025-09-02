import joblib
import numpy as np


def predict_single(sample, model_path='models/ensemble_model.joblib', scaler_path='models/scaler.joblib', features_path='models/selected_features.joblib'):
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
selected = joblib.load(features_path)


sample = np.array(sample).reshape(1, -1)
sample_scaled = scaler.transform(sample)
sample_selected = sample_scaled[:, selected]


pred = model.predict(sample_selected)
return int(pred[0])
