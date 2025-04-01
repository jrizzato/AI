import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def prepare_data(data):
    X = data.drop(columns=['Star type', 'Star category', 'Star color', 'Spectral Class'])
    y = data['Star type']
    
    # Encoding categorical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Load and prepare data
    data = load_data('data/star_classification.csv')
    X, y, label_encoder = prepare_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'model/star_classification_model.pkl')
    joblib.dump(label_encoder, 'model/label_encoder.pkl')

if __name__ == "__main__":
    main()