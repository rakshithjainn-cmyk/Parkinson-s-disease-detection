import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path='data/parkinsons.csv'):
df = pd.read_csv(path)
X = df.drop(columns=['status']) # Assuming 'status' column is label
y = df['status']
return X, y




def preprocess_data(X, y, test_size=0.2, random_state=42):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
return X_train, X_test, y_train, y_test, scaler
