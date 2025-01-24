import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# leer el archivo csv y preprocesamiento de los datos
advertising_df = pd.read_csv('data/Advertising_2023.csv', index_col=0)

# muestra algunas caracteristicas de los datos
# print(advertising_df.head(10)) # muestra las primeras 10 filas
# print()

# advertising_df.info() # muestra la información del dataframe, no necesita print
# print()

# print(advertising_df.describe()) # muestra estadísticas del dataframe
# print()

# print(advertising_df.shape) # muestra la forma del dataframe
# print()

# creamos un dataframe con las características y otro con las etiquetas
x = advertising_df[['digital', 'TV', 'radio', 'newspaper']]
y = advertising_df['sales']

normalized_feature = keras.utils.normalize(x.values)
# print(normalized_feature)
# la normalizacion de los datos es importante para que el modelo pueda aprender de manera más eficiente
# normalizar significa llevar los datos a una escala común, en este caso entre 0 y 1
# cada columna de x se normaliza de manera independiente
# cada columan corresponde a una característica o feature (en este caso digital, TV, radio y newspaper)

# dividir los datos en sets de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# x_trains es el set de entrenamiento de las características. Significa que el modelo se entrenará con estos datos
# x_trains utiliza el 60% de los datos, es decir 719 filas

# x_test es el set de prueba de las características. Significa que el modelo se probará con estos datos
# x_test utiliza el 40% de los datos, es decir 480 filas

# la variable x es el conjunto de características que se utilizarán para PREDECIR las ventas
# consta de 4 columnas: digital, TV, radio y newspaper

# y_train es el set de entrenamiento de las etiquetas. Significa que el modelo se entrenará con estos datos
# y_train utiliza el 60% de los datos, es decir 719 filas

# y_test es el set de prueba de las etiquetas. Significa que el modelo se probará con estos datos
# y_test utiliza el 40% de los datos, es decir 480 filas

# test_size=0.4 significa que el 40% de los datos se usarán para prueba y el 60% para entrenamiento
# random_state=101 es una semilla para que los datos se dividan de la misma manera cada vez que se ejecute el código
# la relacion entre x, y y los sets de entrenamiento y prueba es 60% - 40%

# construimos el modelo
model = keras.models.Sequential()
model.add(keras.layers.Dense(4, activation='relu')) # relu es una función de activación
model.add(keras.layers.Dense(3, activation='relu'))
model.add(keras.layers.Dense(1))

# compilacion (activacion) del modelo agreagando la funcion de perdida y el optimizador
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# entrenamiento (fit) del modelo
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=32)
# epocs es el número de veces que el modelo verá los datos de entrenamiento

# evaluación del modelo
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) # validation loss
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

test_predictions = model.predict(x_test).flatten()
print(test_predictions[:10])
# flatten() convierte la matriz en un array

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

true_values = pd.DataFrame(list(zip(y_test, test_predictions)), columns=['True Values', 'Predictions'])

print(true_values.head(10))

pred_train = model.predict(x_train)
print(np.sqrt(mean_squared_error(y_train, pred_train)))

pred = model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
print()


# ejemplo de uso de la función predict
print('predicción de ventas para digital=0.2, TV=0.1, radio=0.4 y newspaper=0.3')
x_values = pd.DataFrame({
    'digital': [0.2], 
    'TV': [0.1], 
    'radio': [0.4], 
    'newspaper': [0.3]
    })
y_pred = model.predict(x_values)
print(y_pred)

# ejemplo de uso de la función predict
print('predicción de ventas para digital=0.2, TV=0.1, radio=0.4 y newspaper=0.3')
x_values = pd.DataFrame({
    'digital': [0.4], 
    'TV': [0.3], 
    'radio': [0.2], 
    'newspaper': [0.1]
    })
y_pred = model.predict(x_values)
print(y_pred)











