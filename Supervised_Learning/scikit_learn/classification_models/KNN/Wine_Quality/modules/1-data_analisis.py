import pandas as pd

dataset = pd.read_csv("./classification_models/KNN/Wine_Quality/data/winequalityN.csv")

print("Head of the dataset:")
print(dataset.head(), '\n')
print("Description of the dataset:")
print(dataset.describe(), '\n')
print("Info of the dataset:")
print(dataset.info(), '\n')
print("Shape of the dataset:")
print(dataset.shape, '\n')
print("Columns of the dataset:")
print(dataset.columns, '\n')
