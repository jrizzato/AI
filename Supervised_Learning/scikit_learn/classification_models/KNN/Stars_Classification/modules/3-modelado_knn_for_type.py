import joblib

# Cargar las variables preparadas
X_train = joblib.load('./classification_models/KNN/Stars_Classification/data/X_train.pkl')
X_test = joblib.load('./classification_models/KNN/Stars_Classification/data/X_test.pkl')
y_train = joblib.load('./classification_models/KNN/Stars_Classification/data/y_train.pkl')
y_test = joblib.load('./classification_models/KNN/Stars_Classification/data/y_test.pkl')
scaler = joblib.load('./classification_models/KNN/Stars_Classification/data/scaler.pkl')

print("Variables cargadas exitosamente.")

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

training_accuracy = []
test_accuracy = []
k_values = range(1, 30)  # Probar valores de k de 1 a 29, los indices van a ser 0 a 28
k_val_max = 0
knn_score_max = 0
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    knn_score = knn.score(X_test, y_test)
    if knn_score > knn_score_max and k > 3:
        knn_score_max = knn_score
        k_val_max = k
        print(f"Mejor k: {k_val_max} con precisión: {knn_score_max:.2f}")
    test_accuracy.append(knn_score)

knn = KNeighborsClassifier(n_neighbors=k_val_max)
knn.fit(X_train, y_train)
    
# graficamos la precisión del modelo
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(k_values, training_accuracy, label="Precisión en entrenamiento", marker='o')
plt.plot(k_values, test_accuracy, label="Precisión en prueba", marker='o')
plt.title("Precisión del modelo KNN vs Número de vecinos (k)")
plt.xlabel("Número de vecinos (k)")
plt.ylabel("Precisión")
plt.legend()
plt.grid()
plt.show()

# Mapear Star type a Star category
type_to_category = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# creo una estrella de prueba para predecir su tipo
# estrella = [Temperature (K), Luminosity (L/Lo), Radius (R/Ro) y Absolute magnitude (Mv)]
estrella = [[5000, 0.1, 0.5, 10]]
# datos estandarizados de la estrella de prueba
# Aplicar la estandarización
x_estrella_scaled = scaler.transform(estrella)
print("Datos de la estrella antes de la estandarización:")
print(pd.DataFrame(estrella, columns=['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']).head(), '\n')
print("Datos de la estrella después de la estandarización:")
print(pd.DataFrame(x_estrella_scaled, columns=['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']).head(), '\n')
y_estrella_scaled = knn.predict(x_estrella_scaled)
print(f"Predicción de tipo para la estrella de prueba: {y_estrella_scaled[0]} ({type_to_category[y_estrella_scaled[0]]})")

# # Hacer predicciones
y_pred = knn.predict(X_test)

# # Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy:.2f}")

