import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# cargar el dataset
dataset = pd.read_csv("./classification_models/KNN/Wine_Quality/data/winequalityN.csv")

# eliminar filas con valores faltantes
dataset = dataset.dropna()
print(dataset.shape, '\n')

# # codificar la columna 'type' como numérica con label encoding
# # y mostrar la codificación de las clases
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# dataset['type'] = le.fit_transform(dataset['type'])
# print("Codificación de 'type':")
# for clase, valor in zip(le.classes_, range(len(le.classes_))):
#     print(f"{clase}: {valor}")

# Convert the 'quality' score into a **binary classification** (Good: 6+, Bad: <6)
dataset['quality'] = dataset['quality'].apply(lambda x: 1 if x >= 6 else 0)
sns.countplot(hue=dataset['quality'], x=dataset['quality'], legend=False)
plt.title("Distribución de vinos por calidad")
plt.xlabel("Calidad")
plt.ylabel("Cantidad de vinos")
plt.show()

# Separar características (X) y la variable objetivo (y)
X = dataset.drop(['quality','type'], axis=1) # esto es todo el dataset menos la columna 'quality' y 'type'
y = dataset['quality']

# Estandarizar las características (opcional pero recomendado para KNN)
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

import joblib
# Guardar las variables preparadas
joblib.dump(X_train, './classification_models/KNN/Wine_Quality/data/X_train.pkl')
joblib.dump(X_test, './classification_models/KNN/Wine_Quality/data/X_test.pkl')
joblib.dump(y_train, './classification_models/KNN/Wine_Quality/data/y_train.pkl')
joblib.dump(y_test, './classification_models/KNN/Wine_Quality/data/y_test.pkl')
joblib.dump(scaler, './classification_models/KNN/Wine_Quality/data/scaler.pkl')

print("Variables guardadas exitosamente.")