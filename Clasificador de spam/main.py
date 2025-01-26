import nltk
from nltk.corpus import stopwords # importa la lista de stopwords
from nltk.stem import WordNetLemmatizer # importa el lematizador, que es una clase que se encarga de lematizar las palabras
# lematizar es llevar una palabra a su raíz léxica o lema (por ejemplo, "corriendo" a "correr")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np




nltk.download('stopwords') # descarga la lista de stopwords
nltk.download('punkt') # descarga el tokenizador,
nltk.download('punkt_tab')
# tokenizar es dividir un texto en palabras o frases
nltk.download('wordnet') # descarga el lematizador

lemmatizer = WordNetLemmatizer() # crea un objeto de la clase WordNetLemmatizer


# creacion de la representacion de los datos

# cargamos los datos
spam_data = pd.read_csv('data/Spam-Classification.csv')

print('\nDatos Cargados: \n------------------------------------')
print(spam_data.head())

# separamos los datos en datos de entrada y salida
spam_class_bruto = spam_data['CLASS']
spam_messages = spam_data['SMS']
# class spam o ham
# ham se refiere a los mensajes que no son spam, es decir, los mensajes legítimos o deseados

# los mensajes de spam tiene muchos caracteres especiales y números
# por lo que vamos a eliminar esos caracteres y números

def customtokenize(mensaje):
    # separar el mensaje en palabras
    tokens = nltk.word_tokenize(mensaje)
    # siltrar las palabras que no son stopwords
    #nostop = list(filter(lambda token: token not in stopwords.words('english'), tokens))
    nonstop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            nonstop.append(token)
    # lematizar las palabras
    lemmatized = []
    for word in nonstop:
        lemmatized.append(lemmatizer.lemmatize(word))
    return lemmatized

# creamos un objeto de la clase TfidfVectorizer, es decir un objeto que se encarga de convertir los mensajes en vectores (vectorizamos los mensajes)
vectorizer = TfidfVectorizer(tokenizer=customtokenize)

# convertimos los mensajes de spam en vectores
tfidf = vectorizer.fit_transform(spam_messages)

# convertimos TF-IDF a numpy array
tfidf_array = tfidf.toarray()

# construimos un label encoder para las clases spam y ham para convertirlos en valores numéricos
le = preprocessing.LabelEncoder()
spam_class = le.fit_transform(spam_class_bruto)

#convertimos un one-hot encoder vector usando keras
spam_class = tf.keras.utils.to_categorical(spam_class, num_classes=2)

print("TF-IDF matrix shape: ", tfidf_array.shape)
print('One-hot encoded spam class shape: ', spam_class.shape)

# dividimos los datos en datos de entrenamiento y datos de prueba
x_train, x_test, y_train, y_test = train_test_split(tfidf_array, spam_class, test_size=0.1)


# creacion y evaluacion del modelo

# seteamos los hiperparámetros del modelo
NB_CLASSES = 2
N_HIDDEN = 32

model = tf.keras.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
                            input_shape=(x_train.shape[1],),
                            name='Hidden-Layer1',
                            activation='relu'))

model.add(keras.layers.Dense(N_HIDDEN,
                            name='Hidden-Layer2',
                            activation='relu'))

model.add(keras.layers.Dense(NB_CLASSES,
                            name='Output-Layer',
                            activation='softmax')) # softmax es una función de activación 
                                                   # que se utiliza en la capa de salida de una red neuronal 
                                                   # para convertir los valores de la capa de salida en probabilidades

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# el modelo ya fue creado, procedemos al antenamiento del modelo
# seteamos verbose a 1 para que nos muestre el progreso del entrenamiento
VERBOSE = 1
# seteamos los hyperparámetros de entrenamiento
EPOCHS = 10
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2

print('Entrenando el modelo...\n------------------------------------')
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT)
# history es un objeto que contiene la información del entrenamiento del modelo
# contiene los parametros de entrenamiento y validación, como la precisión y la pérdida


# evaluamos el modelo
print('\nPrecision durante el entrenamiento\n------------------------------------')
pd.DataFrame(history.history).plot(figsize=(8, 5))  
# asi se prodrian mostrar las graficas por separado, cada una con un show() correspondiente
# pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["loss"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["val_accuracy"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["val_loss"].plot(figsize=(8, 5))
plt.title('Precision durante el entrenamiento')
plt.show()

print('\nEvaluando el modelo...\n------------------------------------')
model.evaluate(x_test, y_test, verbose=VERBOSE)


# predicciones del modelo

# Predecir para múltiples muestras utilizando procesamiento por lotes

predict_tfidf = vectorizer.transform(['FREE entry to a fun contest',
                                      'Hey, how are you doing?',
                                      'WIN a super prize today',
                                      'do you want a loan?']).toarray()

print('\nPredicciones del modelo\n------------------------------------')
print(predict_tfidf.shape) # el metodo shape nos muestra la forma de la matriz
                           # que puede ser entendida como el número de filas y columnas de la matriz

# preccion usando el modelo
predictions = model.predict(predict_tfidf)
prediction = np.argmax(predictions, axis=1)
print('Prediccon: ', prediction)

# las clases de predicciones
print('Clases de predicciones: ', le.inverse_transform(prediction))