import joblib

# Cargar las variables preparadas
X_train = joblib.load('./classification_models/KNN/Wine_Quality/data/X_train.pkl')
X_test = joblib.load('./classification_models/KNN/Wine_Quality/data/X_test.pkl')
y_train = joblib.load('./classification_models/KNN/Wine_Quality/data/y_train.pkl')
y_test = joblib.load('./classification_models/KNN/Wine_Quality/data/y_test.pkl')

print("Variables cargadas exitosamente.")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Crear y ajustar el modelo de Regresión Logística
log_reg = LogisticRegression(max_iter=1000, random_state=42)  # Puedes ajustar los hiperparámetros
log_reg.fit(X_train, y_train)

# Evaluar el modelo en los conjuntos de entrenamiento y prueba
training_accuracy = log_reg.score(X_train, y_train)
test_accuracy = log_reg.score(X_test, y_test)

print(f"Precisión en entrenamiento: {training_accuracy:.2f}")
print(f"Precisión en prueba: {test_accuracy:.2f}")

# Hacer predicciones
y_pred = log_reg.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de Regresión Logística: {accuracy:.2f}")

training_accuracy = []
test_accuracy = []
C_values = [0.01, 0.1, 1, 10, 100]

for C in C_values:
    log_reg = LogisticRegression(C=C, max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    training_accuracy.append(log_reg.score(X_train, y_train))
    test_accuracy.append(log_reg.score(X_test, y_test))

# Graficar la precisión del modelo
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(C_values, training_accuracy, label="Precisión en entrenamiento", marker='o')
plt.plot(C_values, test_accuracy, label="Precisión en prueba", marker='o')
plt.title("Precisión del modelo de Regresión Logística vs Valor de C")
plt.xlabel("Valor de C (inverso de la regularización)")
plt.ylabel("Precisión")
plt.xscale('log')  # Escala logarítmica para C
plt.legend()
plt.grid()
plt.show()