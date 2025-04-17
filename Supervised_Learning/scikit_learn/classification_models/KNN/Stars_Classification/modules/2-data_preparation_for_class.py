import pandas as pd

# cargar el dataset
dataset = pd.read_csv("./classification_models/KNN/Stars_Classification/data/stars.csv")

# Separar características (X) y la variable objetivo (y)
X = dataset.drop(['Star type','Star category','Star color','Spectral Class'], axis=1) # esto es todo el dataset menos las columnas 'Star type', 'Star category', 'Star color' y 'Spectral Class'
y = dataset['Spectral Class']

from sklearn.preprocessing import LabelEncoder
# Codificar la columna 'Spectral Class'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Ver la codificación
print("Codificación de 'Spectral Class':")
for clase, valor in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{clase}: {valor}")

# Estandarizar las características (opcional pero recomendado para KNN, SVM o regresion logistica)
from sklearn.preprocessing import StandardScaler
# Ver los datos antes de la estandarización
print("Datos antes de la estandarización (primeras 5 filas):")
print(X.head(), '\n')
# Aplicar la estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Ver los datos después de la estandarización
print("Datos después de la estandarización (primeras 5 filas):")
print(pd.DataFrame(X_scaled, columns=X.columns).head())
''' **** Por qué es importante la estandarización en KNN: ****
KNN utiliza distancias (como la distancia euclidiana) para encontrar los vecinos más cercanos.
Si las características tienen escalas diferentes, las características con valores más grandes tendrán un mayor impacto en la distancia, incluso si no son las más relevantes para el problema.
Estandarizar los datos asegura que todas las características contribuyan de manera equitativa al cálculo de las distancias.'''

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=97)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# tamaño del conjunto de entrenamiento y prueba
print("Conjuntos preparados:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Guardar las variables preparadas
import joblib

joblib.dump(X_train, './classification_models/KNN/Stars_Classification/data/X_train.pkl')
joblib.dump(X_test, './classification_models/KNN/Stars_Classification/data/X_test.pkl')
joblib.dump(y_train, './classification_models/KNN/Stars_Classification/data/y_train.pkl')
joblib.dump(y_test, './classification_models/KNN/Stars_Classification/data/y_test.pkl')
joblib.dump(scaler, './classification_models/KNN/Stars_Classification/data/scaler.pkl')
joblib.dump(label_encoder, './classification_models/KNN/Stars_Classification/data/label_encoder.pkl')

print('\n', "Variables guardadas exitosamente.")