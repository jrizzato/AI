import joblib
import pandas as pd

# Cargar las variables preparadas
X_train = joblib.load('./classification_models/KNN/Wine_Quality/data/X_train.pkl')
X_test = joblib.load('./classification_models/KNN/Wine_Quality/data/X_test.pkl')
y_train = joblib.load('./classification_models/KNN/Wine_Quality/data/y_train.pkl')
y_test = joblib.load('./classification_models/KNN/Wine_Quality/data/y_test.pkl')
scaler = joblib.load('./classification_models/KNN/Wine_Quality/data/scaler.pkl')


print("Variables cargadas exitosamente.")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

training_accuracy = []
test_accuracy = []
k_values = range(1, 30)  # Probar valores de k de 1 a 20
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

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

# creo un vino de prueba para predecir su calidad
# vino = [fixed acidity,  volatile acidity,  citric acid,  residual sugar,  chlorides,  free sulfur dioxide,  total sulfur, dioxide  density,    pH,  sulphates,  alcohol]
x_vino = [[7.4, 0.7, 0.22, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
# datos estandarizados del vino de prueba
# Aplicar la estandarización
x_vino_scaled = scaler.transform(x_vino)
print("Datos del vino antes de la estandarización:")
print(pd.DataFrame(x_vino, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']).head(), '\n')
print("Datos del vino después de la estandarización:")
print(pd.DataFrame(x_vino_scaled, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']).head(), '\n')
y_vino = knn.predict(x_vino_scaled)
print(f"Predicción de calidad para el vino de prueba: {y_vino[0]}")

# Hacer predicciones
y_pred = knn.predict(X_test)


# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy:.2f}")
