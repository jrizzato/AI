import nltk # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd # Librería para manipulación de datos
# sklearn: Librería para aprendizaje maquinal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
# TensorFlow y Keras: Librerías para aprendizaje profundo (deep learning)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np # Librería para manipulación de datos


# 1. setup

# Descarga de recursos necesarios de NLTK
nltk.download('stopwords')  # Lista de palabras comunes irrelevantes (stopwords)
nltk.download('punkt')  # Tokenizador para dividir texto en palabras o frases
nltk.download('punkt_tab')
nltk.download('wordnet')  # Recurso para lematización

# Creación de un objeto lematizador
# lematizar es llevar una palabra a su raíz léxica o lema (por ejemplo, "corriendo" a "correr")
lemmatizer = WordNetLemmatizer()


# 2. carga y preprocesamiento de datos

# Cargar datos del archivo CSV
spam_data = pd.read_csv('data/Spam-Classification.csv')
print('\nDatos cargados:\n------------------------------------')
print(spam_data.head())

# Separar datos en entrada (mensajes) y salida (clases)
spam_class_bruto = spam_data['CLASS']
spam_messages = spam_data['SMS']

# Definir una función para preprocesar los mensajes
def customtokenize(mensaje):
    """Tokeniza, elimina stopwords y lematiza un mensaje."""
    # Tokenizar el mensaje
    # tokenizar es dividir el texto en palabras o frases. Es como el split() de Python pero más avanzado
    tokens = nltk.word_tokenize(mensaje)

    # Eliminar stopwords
    # nonstop = [token for token in tokens if token not in stopwords.words('english')]
    nonstop = []
    for token in tokens:
        if token not in stopwords.words('english'):
            nonstop.append(token)

    # Lematizar las palabras
    # lemmatized = [lemmatizer.lemmatize(word) for word in nonstop]
    lemmatized = []
    for word in nonstop:
        lemmatized.append(lemmatizer.lemmatize(word))

    return lemmatized

# Convertir los mensajes en representaciones numéricas utilizando TF-IDF
vectorizer = TfidfVectorizer(tokenizer=customtokenize)
tfidf = vectorizer.fit_transform(spam_messages)  # Matriz TF-IDF
tfidf_array = tfidf.toarray()  # Convertir TF-IDF a un array NumPy

# Codificar las clases (spam y ham) en valores numéricos
le = preprocessing.LabelEncoder()
spam_class = le.fit_transform(spam_class_bruto)
# Convertir las clases a codificación one-hot
spam_class = tf.keras.utils.to_categorical(spam_class, num_classes=2)

print("TF-IDF matrix shape:", tfidf_array.shape)
print("One-hot encoded spam class shape:", spam_class.shape)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(tfidf_array, spam_class, test_size=0.1)


# 3. definición y entrenamiento del modelo

# Definición del modelo de red neuronal
NB_CLASSES = 2  # Número de clases (spam o ham)
N_HIDDEN = 32  # Número de neuronas en las capas ocultas

# model = tf.keras.Sequential([
#     keras.layers.Dense(N_HIDDEN, input_shape=(x_train.shape[1],), activation='relu', name='Hidden-Layer1'),
#     keras.layers.Dense(N_HIDDEN, activation='relu', name='Hidden-Layer2'),
#     keras.layers.Dense(NB_CLASSES, activation='softmax', name='Output-Layer')  # Salida en forma de probabilidades
# ])

model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(N_HIDDEN, input_shape=(x_train.shape[1],), name='Hidden-Layer-1', activation='relu'))
# x_train es la matriz TF-IDF de los mensajes. x_train.shape[0] es el número de mensajes y x_train.shape[1] es el número de características en la matriz TF-IDF
model.add(keras.layers.Dense(N_HIDDEN, name='Hidden-Layer-2', activation='relu'))
model.add(keras.layers.Dense(NB_CLASSES, name='Output-Layer', activation='softmax')) # softmax es una función de activación que convierte las salidas en probabilidades

model.compile(
    loss='categorical_crossentropy',  # Función de pérdida para clasificación multiclase
    metrics=['accuracy']  # Métrica para evaluar el modelo
)

model.summary()  # Resumen de la arquitectura del modelo

# Entrenamiento del modelo
VERBOSE = 1  # Nivel de detalle del proceso de entrenamiento
EPOCHS = 10  # Número de épocas
BATCH_SIZE = 256  # Tamaño de los lotes
VALIDATION_SPLIT = 0.2  # Porcentaje de datos usados para validación

print('Entrenando el modelo...\n------------------------------------')
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

# Visualización del desempeño durante el entrenamiento
print('\nPrecisión durante el entrenamiento\n------------------------------------')
pd.DataFrame(history.history).plot(figsize=(8, 5)) # se muestran las graficas de accuracy y loss en una sola ventana
# asi se prodrian mostrar las graficas por separado, cada una con un show() correspondiente
# pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["loss"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["val_accuracy"].plot(figsize=(8, 5))
# pd.DataFrame(history.history)["val_loss"].plot(figsize=(8, 5))
plt.title('Precisión y pérdida durante el entrenamiento')
plt.show()


# 4. evaluación del modelo y predicciones

# Evaluación del modelo con el conjunto de prueba
print('\nEvaluando el modelo...\n------------------------------------')
model.evaluate(x_test, y_test, verbose=VERBOSE)

# Predicciones con ejemplos personalizados
predict_tfidf = vectorizer.transform([
    'FREE entry to a fun contest',
    'Hey, how are you doing?',
    'WIN a super prize today',
    'do you want a loan?'
]).toarray()

print('\nPredicciones del modelo\n------------------------------------')
print("Shape de la matriz de entrada para predicción:", predict_tfidf.shape)

predictions = model.predict(predict_tfidf)  # Predicciones
prediction = np.argmax(predictions, axis=1)  # Clase predicha
print('Predicción:', prediction)
print('Clases de predicciones:', le.inverse_transform(prediction))
