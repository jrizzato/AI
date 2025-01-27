import pandas as pd # Importamos la librería pandas para trabajar con datos en formato de tabla
import tensorflow as tf, keras
# tensorflow y keras: librerías para construir y entrenar modelos de aprendizaje profundo
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# scikitlearn: librería para realizar tareas de aprendizaje automático
import matplotlib.pyplot as plt

# 1. Cargar los datos
# Leemos el archivo CSV que contiene las características y las especies de las flores de iris
iris_data = pd.read_csv('data/iris.csv')

print("\nDatos originales:\n------------------------------------")
print(iris_data.head(10))

# 2. Codificación de las etiquetas
# Convertimos las etiquetas de las especies en valores numéricos para su uso en el modelo
label_encoder = preprocessing.LabelEncoder()
iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])

print("\nDatos después de codificar las etiquetas:\n------------------------------------")
print(iris_data.head(10))

# Mostramos el mapeo entre las especies y los números asignados
print("\nMapeo de etiquetas y sus valores numéricos:\n------------------------------------")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)

# 3. Separar características (X) y etiquetas (y)
x = iris_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = iris_data['Species']

# 4. Estandarizar las características
# Escalamos las características para que tengan media 0 y varianza 1
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Convertir las etiquetas en una representación one-hot para el modelo
# Esto asegura que cada etiqueta se represente como un vector binario
# Ejemplo: 1 --> [0, 1, 0]
y = tf.keras.utils.to_categorical(y, num_classes=3)

print("\nCaracterísticas escaladas:\n------------------------------------")
print(x[:5, :])
print("\nEtiquetas en formato one-hot:\n------------------------------------")
print(y[:5, :])

# 5. Dividir datos en entrenamiento y prueba
# Usamos un 10% de los datos para pruebas y el resto para entrenamiento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

print("\nDimensiones de los conjuntos de datos:\n------------------------------------")
print(f"Entrenamiento: {x_train.shape}, {y_train.shape}")
print(f"Prueba: {x_test.shape}, {y_test.shape}")

# 6. Construir el modelo de red neuronal
NB_CLASSES = 3  # Número de clases en el conjunto de datos (especies)

# Creamos un modelo secuencial de Keras
model = keras.models.Sequential()

# Agregamos capas densas al modelo
model.add(keras.layers.Dense(128, input_shape=(4,), activation='relu', name='hidden_layer1'))
model.add(keras.layers.Dense(128, activation='relu', name='hidden_layer2'))
model.add(keras.layers.Dense(NB_CLASSES, activation='softmax', name='output_layer'))

# Compilamos el modelo especificando la función de pérdida, optimizador y métrica
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Mostramos un resumen del modelo
model.summary()

# 7. Entrenar el modelo
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.2

# Entrenamos el modelo con los datos de entrenamiento
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=VALIDATION_SPLIT)

# Mostramos la precisión durante el entrenamiento
print("\nPrecisión durante el entrenamiento:\n------------------------------------")
pd.DataFrame(history.history)['accuracy'].plot(figsize=(8, 5))
plt.title("Mejoras en la precisión con las épocas")
plt.show()

# 8. Evaluar el modelo
# Evaluamos el modelo con los datos de prueba para obtener la pérdida y la precisión
print("\nEvaluación del modelo con los datos de prueba:\n------------------------------------")
model.evaluate(x_test, y_test)

# 9. Guardar y cargar el modelo
# Guardamos el modelo en un archivo
model.save('models/iris_model.keras')

# Cargamos el modelo guardado
model = keras.models.load_model('models/iris_model.keras')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 10. Realizar predicciones
# Realizamos una predicción para una nueva muestra de datos
valores_predecir = [[6.6, 3.0, 4.4, 1.4]]
entrada_escalada = scaler.transform(valores_predecir)
prediccion_bruto = model.predict(entrada_escalada)

print("\nPredicción en bruto:\n------------------------------------")
print(prediccion_bruto)

# Determinamos la clase predicha
def obtener_clase_predicha(prediccion_bruto):
    indice = np.argmax(prediccion_bruto)
    return label_encoder.classes_[indice]

clase_predicha = obtener_clase_predicha(prediccion_bruto)
print("\nClase predicha:\n------------------------------------")
print(clase_predicha)

# Otra predicción
valores_predecir = [[1.6, 3.0, 2.4, 3.4]]
entrada_escalada = scaler.transform(valores_predecir)
prediccion_bruto = model.predict(entrada_escalada)
clase_predicha = obtener_clase_predicha(prediccion_bruto)

print("\nOtra predicción:\n------------------------------------")
print(clase_predicha)
