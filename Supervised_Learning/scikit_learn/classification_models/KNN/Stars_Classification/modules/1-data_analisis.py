import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("./classification_models/KNN/Stars_Classification/data/stars.csv")
# https://www.kaggle.com/code/ybifoundation/stars-classification

print("Head of the dataset:")
print(dataset.head(), '\n')
# el valor objetivo (target) es la columna 'Star category' y las variables predictoras son el resto de las columnas

# print("Description of the dataset:")
# print(dataset.describe(), '\n')

print("Info of the dataset:")
print(dataset.info(), '\n')

print("Shape of the dataset:")
print(dataset.shape, '\n')

print("columns of the dataset:")
print(dataset.columns, '\n')

# dataset = dataset.dropna()
# print("Dataset después de eliminar filas con valores faltantes:")
# print(dataset.info(), '\n')
# print(dataset.shape, '\n')
# b = dataset.shape[0]

# print("Cantidad de filas eliminadas:", a - b, '\n')

# print("Columns of the dataset:")
# print(dataset.columns, '\n')

# print("Cantidad de vinos por calidad:")
# print(dataset['quality'].value_counts(), '\n')

# print("Cantidad de vinos por calidad en porcentaje:", '\n')
# print(dataset['quality'].value_counts(normalize=True) * 100)

# plt.figure(figsize=(7, 4))
# # sns.countplot(x=dataset['quality'], palette="coolwarm")

# sns.countplot(hue=dataset['quality'], x=dataset['quality'], legend=False)
# plt.title("Distribución de vinos por calidad")
# plt.xlabel("Calidad")
# plt.ylabel("Cantidad de vinos")
# plt.show()
