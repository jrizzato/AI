import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def clean_data(data):
    # Eliminar filas con valores faltantes
    data = data.dropna()
    return data

def encode_categorical_features(data):
    label_encoders = {}
    categorical_columns = ['Star type', 'Star category', 'Star color', 'Spectral Class']
    
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    return data, label_encoders

def prepare_data(filepath):
    data = load_data(filepath)
    data = clean_data(data)
    data, label_encoders = encode_categorical_features(data)
    
    # Separar características y variable objetivo
    X = data.drop(columns=['Star type'])
    y = data['Star type']
    
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders