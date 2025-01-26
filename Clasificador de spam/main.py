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
                            activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
