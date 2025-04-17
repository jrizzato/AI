import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("./classification_models/KNN/Stars_Classification/data/stars.csv")
# https://www.kaggle.com/code/ybifoundation/stars-classification

print("Head of the dataset:")
print(dataset.head(), '\n')

# hace una descripion estadistica de las columnas numericas
print("Description of the dataset:")
print(dataset.describe(), '\n')

print("Info of the dataset:")
print(dataset.info(), '\n')

print("Shape of the dataset:")
print(dataset.shape, '\n')

print("columns of the dataset:")
print(dataset.columns, '\n')

print(dataset[['Star type']].value_counts(), '\n') # numeros del 0 al 5 (de todas hay 40)
print(dataset[['Star category']].value_counts(), '\n') # texto: Brown Dwarf, Hypergiant, Main Sequence, etc (de todas hay 40)
# ademas, cada Star type corresponde con un Star category, por lo que no es necesario tener ambas columnas
print(dataset[['Star type', 'Star category']].value_counts(), '\n') 

print(dataset[['Star color']].value_counts(), '\n') # texto: Red, Blue, Blue-White, etc (distrubucion variada)

print(dataset[['Spectral Class']].value_counts(), '\n') # texto: O, B, A, F, G, K, M (distribucion variada)

# el info dice que no columas not-null, asi que no hay valores faltantes, y no hace falta hacer dropna
# dataset = dataset.dropna()
# print("Dataset después de eliminar filas con valores faltantes:")
# print(dataset.info(), '\n')
# print(dataset.shape, '\n')

# print("Cantidad de filas eliminadas:", a - b, '\n')

'''
Voy a hacer dos predicciones:
1. Clasificación de estrellas por tipo (Star type) porque es la que tiene menos clases (6), estan uniformemente distribuidas, 
   y ya son valores numericos
2. Clasificación de estrellas por clase espectral (Spectral Class)

'''


plt.figure(figsize=(7, 4))

sns.countplot(hue=dataset['Star type'], x=dataset['Star type'], legend=False)
plt.title("Distribución de estrellas por tipo")
plt.xlabel("Tipo de estrella")
plt.ylabel("Cantidad de estrellas")
plt.show()

sns.countplot(hue=dataset['Star category'], x=dataset['Star category'], legend=False)
plt.title("Distribución de estrellas por categoria")
plt.xlabel("Categoria de estrella")
plt.ylabel("Cantidad de estrellas")
plt.show()

sns.countplot(hue=dataset['Star color'], x=dataset['Star color'], legend=False)
plt.title("Distribución de estrellas por color")
plt.xlabel("Color de estrella")
plt.ylabel("Cantidad de estrellas")
plt.show()

sns.countplot(hue=dataset['Spectral Class'], x=dataset['Spectral Class'], legend=False)
plt.title("Distribución de estrellas por clase espectral")
plt.xlabel("Clase espectral")
plt.ylabel("Cantidad de estrellas")
plt.show()
