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

# construimos el modelo corregido
model = keras.models.Sequential()
model.add(keras.layers.Dense(4, activation='relu')) 
model.add(keras.layers.Dense(7, activation='relu')) 
model.add(keras.layers.Dense(6, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# corregimos el modelo con early stopping (es)
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=2000, verbose=0, callbacks=[es])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) 
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()



