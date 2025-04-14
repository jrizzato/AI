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

print("------------------------------------------------------")
# # Seleccionar dos características (por ejemplo, 'alcohol' y 'pH')
# X_train_2d = X_train[:, [9, 8]]  # Índices de las columnas seleccionadas
# X_test_2d = X_test[:, [9, 8]]

# # Entrenar el modelo KNN con las dos características seleccionadas
# knn_2d = KNeighborsClassifier(n_neighbors=k_val_max)
# knn_2d.fit(X_train_2d, y_train)

# # Graficar las fronteras de decisión
# import mglearn
# plt.figure(figsize=(10, 6))
# mglearn.plots.plot_2d_separator(knn_2d, X_train_2d, fill=True, alpha=0.3)
# mglearn.discrete_scatter(X_train_2d[:, 0], X_train_2d[:, 1], y_train)
# plt.title(f"Fronteras de decisión del modelo KNN (k={k_val_max})")
# plt.xlabel("Alcohol")
# plt.ylabel("pH")
# plt.legend(["Clase 0", "Clase 1", "Clase 2"], loc="best")
# plt.grid()
# plt.show()