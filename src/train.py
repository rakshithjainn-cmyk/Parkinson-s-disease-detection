import os
import joblib
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import load_data, preprocess_data
from src.feature_engineering import select_features
from src.models import build_models


def train_and_evaluate():
os.makedirs('models', exist_ok=True)


X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
X_train_fs, selected = select_features(X_train, y_train, k=10)
X_test_fs = X_test[:, selected]


_, _, _, ensemble = build_models()
ensemble.fit(X_train_fs, y_train)


preds = ensemble.predict(X_test_fs)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))


joblib.dump(ensemble, 'models/ensemble_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(selected, 'models/selected_features.joblib')
print("Models saved in 'models/' directory.")


if __name__ == '__main__':
train_and_evaluate()
