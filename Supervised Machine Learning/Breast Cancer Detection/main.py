import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time


# Exploración de datos del dataset (dataset exploration)
# ===============================================

# cargamos el archivo csv usando la librería pandas
dataset = pd.read_csv('data/data.csv')
# revisamos el dataset
print('dataset head:\n', dataset.head(), '\n')


# separamos los datos (data) de los valores objetivos (etiquetas, targets)

# obtenemos solamente la columna diagnostico (diagnosis) que es la que queremos predecir
targets = dataset['diagnosis'] # M o B
# target = dataset.diagnosis # asi tambien funciona
# revisamos los targets
print('targets:\n', targets.head(20), '\n')
print('targets shape:\n', targets.shape, '\n')
# eliminamos la columna diagnosis del dataset, junto con otras que no necesitamos
data = dataset.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
# revisamos los datos
print('data:\n', data.head(20), '\n')
print('data shape:\n', data.shape, '\n')


# verificando si el dataset esta balanceado

# contamos cuantos valores de cada tipo hay en la columna diagnosis
ax = sns.countplot(x=targets, label='Conteo')
B, M = targets.value_counts()
print('Numero de tumores benignos: ', B)
print('Numero de tumores malignos: ', M)
plt.show()


# verificamos la correlación entre las variables

#Setting style for the Seaborn graph
sns.set(style="whitegrid", palette="muted")
#Getting targets
data_dia = targets
#data normalization for plotting
data_n_2 = (data - data.mean()) / (data.std())

#Getting only first 10 features
data_vis = pd.concat([targets,data_n_2.iloc[:,0:10]],axis=1)
#Flattening the dataset
data_vis = pd.melt(data_vis,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
# sns.stripplot(x="features", y="value", hue="diagnosis", data=data_vis) # otra opcion
plt.xticks(rotation=90)
plt.show()

#Getting all features from 10th to 20th
data_vis = pd.concat([targets,data_n_2.iloc[:,10:20]],axis=1)
data_vis = pd.melt(data_vis,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
plt.xticks(rotation=90)
plt.show()

#Getting the last 10 features in the dataset
data_vis = pd.concat([targets,data_n_2.iloc[:,20:30]],axis=1)
data_vis = pd.melt(data_vis,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_vis)
plt.xticks(rotation=90)
plt.show()


# Preprocesamiento de datos del dataset (dataset preprocessing)
# ===============================================