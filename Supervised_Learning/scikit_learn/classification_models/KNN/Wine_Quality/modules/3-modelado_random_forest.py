import joblib

# Cargar las variables preparadas
X_train = joblib.load('./classification_models/KNN/Wine_Quality/data/X_train.pkl')
X_test = joblib.load('./classification_models/KNN/Wine_Quality/data/X_test.pkl')
y_train = joblib.load('./classification_models/KNN/Wine_Quality/data/y_train.pkl')
y_test = joblib.load('./classification_models/KNN/Wine_Quality/data/y_test.pkl')

print("Variables cargadas exitosamente.")

# Importar la clase RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

training_accuracy = []
test_accuracy = []
n_estimators_values = range(1, 200, 20)
n_val_max = 0
rf_score_max = 0
for n in n_estimators_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=97)
    rf.fit(X_train, y_train)
    training_accuracy.append(rf.score(X_train, y_train))
    test_accuracy.append(rf.score(X_test, y_test))
    rand_score = rf.score(X_test, y_test)
    if rand_score > rf_score_max and n > 3:
        rf_score_max = rand_score
        n_val_max = n
        for max_features in ['sqrt', 'log2', None, 2, 3]:
            rf = RandomForestClassifier(n_estimators=n_val_max, max_features=max_features, random_state=97)
            rf.fit(X_train, y_train)
            accuracy = rf.score(X_test, y_test)
            print(f"Mejor n_estimators: {n_val_max} con precisión: {rf_score_max:.5f} y max_features={max_features}")
    
# ajustamos el modelo con el mejor n_estimators
rf = RandomForestClassifier(n_estimators=n_val_max, random_state=97)
rf.fit(X_train, y_train)
print(f"Mejor n_estimators: {n_val_max} con precisión: {rf_score_max:.5f}")

# Evaluar el modelo en los conjuntos de entrenamiento y prueba
training_accuracy_single = rf.score(X_train, y_train)
test_accuracy_single = rf.score(X_test, y_test)

print(f"Precisión en entrenamiento: {training_accuracy_single:.5f}")
print(f"Precisión en prueba: {test_accuracy_single:.5f}")

# Hacer predicciones
y_pred = rf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo Random Forest: {accuracy:.5f}")

# Graficar la precisión del modelo
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, training_accuracy, label="Precisión en entrenamiento", marker='o')
plt.plot(n_estimators_values, test_accuracy, label="Precisión en prueba", marker='o')
plt.title("Precisión del modelo Random Forest vs Número de árboles")
plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Precisión")
plt.legend()
plt.grid()
plt.show()

import numpy as np

# Obtener las importancias de las características
importances = rf.feature_importances_

# Obtener los nombres de las características (si están disponibles)
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]

# Ordenar las características por importancia
indices = np.argsort(importances)[::-1]

# Graficar
plt.figure(figsize=(10, 6))
plt.title("Importancia de las características")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), np.array(feature_names)[indices], rotation=45, ha="right")
plt.xlabel("Características")
plt.ylabel("Importancia")
plt.tight_layout()
plt.show()