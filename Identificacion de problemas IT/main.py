import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 1 cargamos los datos
# el archivo con tiene 7 features (caracteristicas) y 3 target (objetivos - root causes)
# features: CPU_LOAD, MEMORY_LEAK_LOAD, DELAY, ERROR_1000, ERROR_1001, ERROR_1002, ERROR_1003
# target: DATABASE_ISSUE, MEMORY_LEAK, NETWORK_DELAY
issues_data = pd.read_csv('data/root_cause_analysis.csv')

print('\nMuestra de datos:\n----------------')
print(issues_data.head())


# 2 codificacion de las root causes
# codificamos las root causes en un formato numerico
etiqueta_encoder = preprocessing.LabelEncoder()
issues_data['ROOT_CAUSE'] = etiqueta_encoder.fit_transform(issues_data['ROOT_CAUSE'])

print('\nCodificacion de las root causes:\n----------------')
print(issues_data.head(10))

# La codificacion de las root causes es:
# DATABASE_ISSUE -> 0
# MEMORY_LEAK -> 1
# NETWORK_DELAY -> 2
# porque se hace por orden alfabetico

# verificamos las etiquetas
print('\nEtiquetas de las root causes:\n----------------')
# Definimos las root causes
root_causes = ['DATABASE_ISSUE', 'MEMORY_LEAK', 'NETWORK_DELAY']
# Ajustamos y transformamos las root causes
encoded_labels = etiqueta_encoder.fit_transform(root_causes)
# Mostramos la codificación
for cause, code in zip(root_causes, encoded_labels):
    print(f'{cause} -> {code}')

# Otra forma, mostramos el mapeo entre las especies y los números asignados
print("\nMapeo de etiquetas y sus valores numéricos:\n------------------------------------")
for i, item in enumerate(etiqueta_encoder.classes_):
    print(item, '-->', i)

# 3 separamos caracteristicas (x) y etiquetas (y)
x = issues_data[['CPU_LOAD', 'MEMORY_LEAK_LOAD', 'DELAY', 'ERROR_1000', 'ERROR_1001', 'ERROR_1002', 'ERROR_1003']]
y = issues_data['ROOT_CAUSE']

print("\nCaracterísticas:\n------------------------------------")
print(x.head())
print("\nEtiquetas:\n------------------------------------")
print(y.head())

# 4 dividimos los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("\nDimensiones de los conjuntos de datos:\n------------------------------------")
print(f"Entrenamiento: {x_train.shape}, {y_train.shape}")
print(f"Prueba: {x_test.shape}, {y_test.shape}")

# 5 creamos el modelo
model = keras.Sequential()
model.add(keras.layers.Dense(32, input_dim=7, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# 6 entrenamiento (fit) del modelo
history = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.2, verbose=1)

print('\nHistorial de entrenamiento:\n----------------')
print(history.history.keys())

# evaluación del modelo
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


#7 evalamos el modelo con los datos de prueba
# Evaluamos el modelo con los datos de prueba para obtener la pérdida y la precisión
print("\nEvaluación del modelo con los datos de prueba:\n------------------------------------")
model.evaluate(x_test, y_test)

# 8 predicciones
# Hacemos predicciones con los datos de prueba
print("\nPredicciones:\n------------------------------------")
test_predictions = model.predict(x_test).flatten()
print(test_predictions[:10])

# ejemplo de uso de la función predict
print('predicción de ventas para cpu_load=0, memory_leak_load=0, delay=1, error_1000=1, error_1001=0, error_1002=0, error_1003=0')
x_values = pd.DataFrame({
    'CPU_LOAD': [0],
    'MEMORY_LEAK_LOAD': [0],
    'DELAY': [1],
    'ERROR_1000': [1],
    'ERROR_1001': [0],
    'ERROR_1002': [0],
    'ERROR_1003': [0]
    })
# y_pred = np.argmax(model.predict(x_values), axis=1) # si lo hago de una no funciona classification_report
# print(etiqueta_encoder.inverse_transform(y_pred))
predicciones = model.predict(x_test)
y_pred = np.argmax(predicciones, axis=1)

print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=root_causes))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=root_causes, yticklabels=root_causes)
plt.title('Matriz de confusión')
plt.ylabel('Etiqueta verdadera')
plt.xlabel('Etiqueta predicha')
plt.show()

