import pandas as pd
import os
import tensorflow as tf, keras
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# cargamos la informacion de los datos y la revisamos (preprocesamiento)
iris_data = pd.read_csv('data/iris.csv')

print("\nDatos con tu etiqueta:\n------------------------------------")
print(iris_data.head(10))

# Usar un codificador de etiquetas para convertir cadenas a valores numéricos
# para la variable objetivo

# from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() # sirve para convertir las etiquetas de las clases en valores numéricos
iris_data['Species'] = label_encoder.fit_transform(iris_data['Species']) # convierte las etiquetas de las clases en valores numéricos

print("\n\nDatos despues de la codificacion de la etiqueta:\n------------------------------------")
print(iris_data.head(10))

# mostrar que numero se le asignó a cada especie
print("\n\nMapeo de la etiqueta y su numero:\n------------------------------------")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)

# dividir los datos en caracteristicas y etiquetas
x = iris_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
y = iris_data['Species']

# estandarizacion de los datos
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# StandardScaler es una clase de la biblioteca sklearn que se utiliza para estandarizar características 
# eliminando la media y escalando a la varianza unitaria.
# El método .fit(X_data) calcula la media y la desviación estándar de X_data y las almacena en el objeto scaler.

# Convertir la variable objetivo a un array de codificación one-hot
y = tf.keras.utils.to_categorical(y, num_classes=3)

print("\nFeatures after scaling :\n------------------------------------")
print(x[:5,:])
print("\nTarget after one-hot-encoding :\n------------------------------------")
print(y[:5,:])

# dividir los datos en sets de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

print("\nTrain Test Dimensions:\n------------------------------------")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# construir el modelo

# numero de clases en la variable objetivo
NB_CLASSES=3

# creamos un modelo secuencial
model = keras.models.Sequential()

# agregamos una capa densa con 128 neuronas (perceptrones) y función de activación relu
model.add(keras.layers.Dense(128,
                            input_shape=(4,), # 4 variables de entrada (features)
                            name='hidden_layer1',
                            activation='relu')) # función de activación relu (rectified linear unit)

# agregamos una capa densa con 128 neuronas y función de activación relu
model.add(keras.layers.Dense(128,
                            name='hidden_layer2',
                            activation='relu')) # función de activación relu

# agregamos una capa densa con 3 neuronas (una para cada clase) y función de activación softmax
model.add(keras.layers.Dense(NB_CLASSES,
                            name='output_layer',
                            activation='softmax')) # función de activación softmax

# compilamos el modelo
model.compile(loss='categorical_crossentropy', # función de pérdida
              optimizer='adam', # optimizador
              metrics=['accuracy']) # métrica de evaluación

model.summary()


# entrenamos el modelo

# ponemos el verbose en 1 para que muestre el progreso del entrenamiento
VERBOSE = 1

# configuramos los hyperparámetros

# numero de lotes
BATCH_SIZE = 16
# numero de epochs
EPOCHS = 10
# configuramos el tamaño de validación al 20%
VALIDATION_SPLIT = 0.2

# entrenamos el modelo
# Ajustar el modelo. Esto realizará todo el ciclo de entrenamiento, incluyendo
# propagación hacia adelante (forward propagation), 
# cálculo de pérdida (loss computation), 
# propagación hacia atrás (backward propagation)
# y descenso de gradiente (gradient descent).
# Ejecutar para los tamaños de lote y epoch especificados
# Realizar validación después de cada epoch
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)

print("\nPrecisión durante el entrenamiento:\n------------------------------------")
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")
plt.show()

print("\nEvaluacion contra el dataset:\n------------------------------------")
model.evaluate(x_test,y_test)
# evaluate devuelve la pérdida y las métricas de evaluación del modelo en el conjunto de datos de prueba

# guardar el modelo
model.save('models/iris_model.keras')
# cargar un modelo
model = keras.models.load_model('models/iris_model.keras')
# recompilar el modelo con las métricas deseadas
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# mostrar resumen del modelo cargado
model.summary()


# hacer predicciones con el deep learning model
# datos de la predicción
valores_predecir = [[6.6, 3.0, 4.4, 1.4]] # valores de las características, son las medidas características de una flor de iris
# escalado de los datos
entrada_escalada = scaler.transform(valores_predecir)
# predicción en bruto
prediccion_bruto = model.predict(entrada_escalada)
# predicción en bruto
print("\nPredicción en bruto:\n------------------------------------")
print(prediccion_bruto)
# prediccion de la clase
prediccion_clase = np.argmax(prediccion_bruto)
print(type(prediccion_clase))
# argmax devuelve el índice del valor máximo a lo largo de un eje
# el resultado de argmax es la clase predicha, por ejemplo prediccion en bruto devuelve
# [[0.001, 0.9, 0.099]] entonces el argmax de eso es 1
# la clase 1 es la clase predicha, que este caso es versicolor
# clase predicha
print("\nClase predicha:\n------------------------------------")
print(label_encoder.classes_[prediccion_clase]) # muestra la clase predicha - versicolor

prediccion_clase = np.int64(2)
print("\nClase predicha:\n------------------------------------")
print(label_encoder.classes_[prediccion_clase]) # muestra la clase predicha - virginica

# hacer predicciones con el deep learning model
valores_predecir = [[1.6, 3.0, 2.4, 3.4]]
entrada_escalada = scaler.transform(valores_predecir)
prediccion_bruto = model.predict(entrada_escalada) # <--- aca se hace la predicción
prediccion_clase = np.argmax(prediccion_bruto)
print(label_encoder.classes_[prediccion_clase])