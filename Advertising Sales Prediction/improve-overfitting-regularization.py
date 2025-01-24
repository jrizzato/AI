import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

advertising_df = pd.read_csv('data/Advertising_2023.csv', index_col=0)

x = advertising_df[['digital', 'TV', 'radio', 'newspaper']]
y = advertising_df['sales']

normalized_feature = keras.utils.normalize(x.values)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# construimos el modelo corregido con la tectica de regularization
model = keras.models.Sequential()
model.add(keras.layers.Dense(4, input_dim=4, activation='relu')) 
model.add(keras.layers.Dense(4, activation='relu')) 
model.add(keras.layers.Dense(4, activation='relu'))
model.add(keras.layers.Dense(56, input_dim=56, kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(keras.layers.Dense(1))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=100)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()



