import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time


# Exploración de datos del dataset (dataset exploration)
# ===============================================
# ¿cual es la finalidad de la exploracion de los datos?
# - entender los datos
# - verificar la calidad de los datos
# - identificar patrones en los datos
# - determinar la relación entre las variables
# - determinar la importancia de las variables
# - determinar la distribución de las variables

# cargamos el archivo csv usando la librería pandas
dataset = pd.read_csv('data/data.csv')
# revisamos el dataset
print('dataset head:\n', dataset.head(), '\n')
# ¿que observamos con el metodo head?
# - que el dataset tiene 33 columnas
# - que la primer columna es un id
# - que la segunda columna es el diagnostico (M o B)
# - que las siguientes columnas son valores numéricos
# - que la última columna es NaN


# separamos los datos (data) de los valores objetivos (etiquetas, targets)

# obtenemos solamente la columna diagnostico (diagnosis) que es la que queremos predecir
targets = dataset['diagnosis'] # M o B
# target = dataset.diagnosis # asi tambien funciona
# revisamos los targets
print('targets:\n', targets.head(20), '\n')
print('targets shape:\n', targets.shape, '\n')
# ¿que observamos con el metodo shape?
# - que hay 569 filas
# - que hay 1 columna
# eliminamos la columna diagnosis del dataset, junto con otras que no necesitamos
data = dataset.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
# ¿de donde sacamos que las columnas que queremos eliminar se llaman asi?
# - del metodo head
# ¿porque eliminamos la columna diagnosis?
# - porque es la columna que queremos predecir
# ¿porque eliminamos la columna id?
# - porque no aporta información relevante
# ¿porque eliminamos la columna Unnamed: 32?
# - porque es NaN
# revisamos los datos
print('data:\n', data.head(20), '\n')
# ¿que observamos con el metodo head?
# - que hay 569 filas
# - que hay 30 columnas (antes habia 33 pero eliminamos 3)
# revisamos la forma de los datos
print('data shape:\n', data.shape, '\n')


# verificando si el dataset esta balanceado
# ¿para que queremos saber si el dataset esta balanceado?
# - para saber si hay la (mas o menos) misma cantidad de muestras para cada clase
# - para saber si hay una clase que tiene mas muestras que otra
# - para saber si hay una clase que tiene menos muestras que otra


# contamos cuantos valores de cada tipo hay en la columna diagnosis
ax = sns.countplot(x=targets, label='Conteo')
B, M = targets.value_counts()
print('Numero de tumores benignos: ', B)
print('Numero de tumores malignos: ', M)
plt.show()


# verificamos la correlación entre las variables
# ¿que significa la correlacion entre las variables? ¿porque es importante?
# - la correlación entre las variables indica si las variables están relacionadas entre sí
# - es importante porque nos permite saber si una variable afecta a otra

#Setting style for the Seaborn graph
sns.set(style="whitegrid", palette="muted")
#Getting targets
data_dia = targets
#data normalization for plotting
data_n_2 = (data - data.mean()) / (data.std())

#Getting only first 10 features
data_vis = pd.concat([targets,data_n_2.iloc[:,0:10]],axis=1)
# ¿que hace el metodo concat?
# - concatena dos dataframes, targets y data_n_2.iloc[:,0:10]
# - axis=1 indica que se concatenan por columnas
# - axis=0 indica que se concatenan por filas
# - en este caso, se concatenan por columnas
# - el resultado es un dataframe con 569 filas y 11 columnas
# ¿que hace el metodo iloc?
# - selecciona un rango de columnas
# - en este caso, selecciona las columnas de 0 a 9
#Flattening the dataset
# ¿que significa "flattening the dataset"?
# - significa que estamos convirtiendo el dataset en un formato que se puede visualizar
# - en este caso, estamos convirtiendo el dataset en un formato que se puede visualizar con seaborn
# - el formato que se puede visualizar con seaborn es un formato largo
# - el formato largo es un formato en el que cada fila tiene una sola observación
# data_vis = pd.melt(data_vis,id_vars="diagnosis",
#                     var_name="features",
#                     value_name='value')
# plt.figure(figsize=(10,10))
# tic = time.time()
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
# # sns.stripplot(x="features", y="value", hue="diagnosis", data=data_vis) # otra opcion
# plt.xticks(rotation=90)
# plt.show()

#Getting all features from 10th to 20th
# data_vis = pd.concat([targets,data_n_2.iloc[:,10:20]],axis=1)
# data_vis = pd.melt(data_vis,id_vars="diagnosis",
#                     var_name="features",
#                     value_name='value')
# plt.figure(figsize=(10,10))
# tic = time.time()
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
# plt.xticks(rotation=90)
# plt.show()

#Getting the last 10 features in the dataset
# data_vis = pd.concat([targets,data_n_2.iloc[:,20:30]],axis=1)
# data_vis = pd.melt(data_vis,id_vars="diagnosis",
#                     var_name="features",
#                     value_name='value')
# plt.figure(figsize=(10,10))
# tic = time.time()
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
# plt.xticks(rotation=90)
# plt.show()


# Preprocesamiento de datos del dataset (dataset preprocessing)
# ===============================================

# ¿cual es la finalidad del preprocesamiento de los datos?
# - limpiar los datos
# - transformar los datos
# - normalizar los datos
# - estandarizar los datos
# - reducir la dimensionalidad de los datos

dataset = pd.read_csv('data/data.csv')
targets = dataset['diagnosis']
targets.value_counts()

# B = 357
# M = 212  


# Normalización de los valores objetivos (targets)

# ¿porque normalizamos los valores objetivos?
# - para que los valores objetivos tengan la misma escala
# - para que los valores objetivos sean más fáciles de comparar
# - para que los valores objetivos sean más fáciles de visualizar

# para normalizar los datos necesitamos numeros, pero los valores objetivos son strings
# Cambiamos los valores objetivos de string a numeros
targets = targets.map({'M':1, 'B':0})
# ¿que hace el metodo map?
# - mapea los valores de una serie a otros valores
# - en este caso, mapea 'M' a 1 y 'B' a 0
# targets = np.where(targets == 'M', 1, 0) # asi tambien funciona
# revisamos los targets
print('targets:\n', targets.head(20), '\n')  

print('Dataset head:\n', dataset.head(), '\n') #33 columnas
lista = ['Unnamed: 32', 'id', 'diagnosis']
data = dataset.drop(lista, axis = 1 )
# ¿porque eliminamos las columnas Unnamed: 32, id y diagnosis?
# - porque no aportan información relevante
# revisamos los datos
print('data:\n', data.head(20), '\n') #30 columnas
print('data shape:\n', data.shape, '\n')


# normalizacion de los datos (features)

# ¿que es la normalización de los datos?
# - es el proceso de escalar los datos a un rango fijo
# - el rango fijo es generalmente de 0 a 1
# - la normalización de los datos es importante para que los datos tengan la misma escala
# - significa que la desviación estándar de los datos es 1
# - la media de los datos es 0
# - los datos normalizados son más fáciles de comparar

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# ¿que hace el metodo fit_transform?
# - ajusta el escalador a los datos y transforma los datos
# revisamos los datos
print('data normalizado:\n', data, '\n')
# ¿que observamos con el metodo head?
# - que los datos estan normalizados
# - que los datos tienen la misma escala
# - que los datos son más fáciles de comparar
# - que los datos son más fáciles de visualizar

# separar los datos en datos de entrenamiento y datos de prueba
# ¿porque separamos los datos en datos de entrenamiento y datos de prueba?
# - para entrenar el modelo con los datos de entrenamiento
# - para probar el modelo con los datos de prueba

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.1, random_state=42)
# ¿que hace el metodo train_test_split?
# - divide los datos en datos de entrenamiento y datos de prueba
# - el tamaño de los datos de prueba es 20%
# - el tamaño de los datos de entrenamiento es 80%
# - el estado aleatorio es 42
# ¿que significa el estado aleatorio?
# - significa que los datos se dividen de la misma manera cada vez que se ejecuta el código
# revisamos los datos
print('X_train:\n', X_train, '\n')
print('X_test:\n', X_test, '\n')
print('y_train:\n', y_train, '\n')
print('y_test:\n', y_test, '\n')
print('X_test shape:', X_test.shape, '\n')
print('X_train shape:', X_train.shape, '\n')
print('y_test shape:', y_test.shape, '\n')
print('y_train shape:', y_train.shape, '\n')


# Entrenamiento del modelo (model training)

# KNN (K-Nearest Neighbors)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
# ¿que es el clasificador K-Nearest Neighbors?
# - es un algoritmo de aprendizaje supervisado
# - es un algoritmo de clasificación

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [5, 7, 11, 15, 3],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size': [10, 30, 50, 100],
              }

grid_search = GridSearchCV(knn, param_grid=param_grid)
# el primer argumento de GridSearchCV es el algoritmo que queremos optimizar
# el segundo argumento de GridSearchCV es el espacio de búsqueda
# entrenamos la grilla de búsqueda
grid_search.fit(X_train, y_train)

# obtener los mejores paramatros
print(grid_search.best_params_)
print(grid_search.best_estimator_)

# definimos el knn con los mejores parametros
knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=5)
# ¿porque definimos el knn con los mejores parametros?
# - para que el knn tenga el mejor rendimiento

# entrenamos el knn
knn.fit(X_train, y_train)


# Evaluacion del modelo

# Accuracy (exactitud)
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:', accuracy)
# ¿que es la exactitud?
# - es el porcentaje de predicciones correctas
# - es una medida de rendimiento del modelo
# - es una medida de la calidad del modelo

# matriz de confusion (confusion matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix:\n', cm)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
# ¿que es la matriz de confusion?
# - es una tabla que muestra las predicciones correctas e incorrectas
# - es una tabla que muestra las predicciones positivas y negativas
# - es una tabla que muestra los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos
# verdaderos positivos (TP): predicciones positivas correctas. Por ejemplo, tumores malignos que se predicen como malignos
# falsos positivos (FP): predicciones positivas incorrectas. Por ejemplo, tumores benignos que se predicen como malignos
# verdaderos negativos (TN): predicciones negativas correctas. Por ejemplo, tumores benignos que se predicen como benignos
# falsos negativos (FN): predicciones negativas incorrectas. Por ejemplo, tumores malignos que se predicen como benignos

# Precision (precision)
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('precision:', precision)
# ¿que es la precision?
# - es el porcentaje de predicciones positivas correctas

# Recall (sensibilidad)
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('recall:', recall)

