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

# Crear y ajustar el modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Puedes ajustar los hiperparámetros
rf.fit(X_train, y_train)

# Evaluar el modelo en los conjuntos de entrenamiento y prueba
training_accuracy = rf.score(X_train, y_train)
test_accuracy = rf.score(X_test, y_test)

print(f"Precisión en entrenamiento: {training_accuracy:.2f}")
print(f"Precisión en prueba: {test_accuracy:.2f}")

# Hacer predicciones
y_pred = rf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo Random Forest: {accuracy:.2f}")

training_accuracy = []
test_accuracy = []
n_estimators_values = [10, 50, 100, 200]

for n in n_estimators_values:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    training_accuracy.append(rf.score(X_train, y_train))
    test_accuracy.append(rf.score(X_test, y_test))

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