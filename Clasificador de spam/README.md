# Spam Classifier with Deep Learning y Neural Network

Este proyecto implementa un clasificador de mensajes spam utilizando aprendizaje profundo (deep learning) y redes neuronales automáticas (ANN). A través de la representación TF-IDF de los mensajes y una red neuronal densa, el modelo es capaz de distinguir entre mensajes spam y no spam (ham).

## Descripción

El proyecto utiliza varias etapas para preprocesar los datos y construir un modelo de clasificación:

- **Preprocesamiento de texto:** Eliminación de stopwords, lematización y tokenización utilizando NLTK.
- **Representación de datos:** Conversión de los mensajes en vectores TF-IDF.
- **Construcción del modelo:** Creación de una red neuronal con TensorFlow y Keras para clasificar los mensajes.
- **Entrenamiento y evaluación:** Uso de datos de entrenamiento y prueba para entrenar y evaluar el rendimiento del modelo.
- **Predicciones personalizadas:** Clasificación de nuevos mensajes ingresados por el usuario.

### Características principales:

- Procesamiento de texto: tokenización, eliminación de stopwords y lematización.
- Representación de texto mediante TF-IDF.
- Modelo de red neuronal multicapa con activación ReLU y softmax.
- Evaluación del modelo y predicciones personalizadas.

### Propósito:

Facilitar la identificación de mensajes de spam en aplicaciones de mensajería mediante un enfoque de aprendizaje profundo.

### Prerrequisitos 🗉

Para ejecutar este proyecto, necesitas instalar las siguientes herramientas:

- **Sistema Operativo:** Cualquier sistema compatible con Python (Windows, Linux, macOS).
- **Python:** Versión 3.8 o superior.
- **Librerías requeridas:**
  - `nltk`
  - `pandas`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`
  - `numpy`

Instala las dependencias con:
```bash
pip install -r deps/requirements.txt
```

## Arquitectura del modelo 🏛️

El modelo de red neuronal tiene la siguiente arquitectura:

- **Capa 1:** Densa (fully connected) con 32 neuronas, activación ReLU, y toma como entrada la matriz TF-IDF.
- **Capa 2:** Densa con 32 neuronas y activación ReLU.
- **Capa de salida:** Densa con 2 neuronas (una por cada clase: Spam y Ham), activación softmax.

Diagrama de capas:

```
Input Layer -> Hidden Layer 1 (32 neuronas, ReLU) -> Hidden Layer 2 (32 neuronas, ReLU) -> Output Layer (2 neuronas, softmax)
```

El modelo está entrenado utilizando la función de pérdida *categorical_crossentropy* y optimizado para la métrica de precisión.

## Parámetros de Entrenamiento

- Tamaño del Lote (Batch Size): 256
- Épocas: 10
- Función de Pérdida: Entropía Cruzada Categórica
- Métricas: Precisión (Accuracy)
- División de Validación: 0.2
- Optimizador: Por defecto (Adam)

## Ejecutando las Pruebas ⚙️

El conjunto de pruebas utiliza datos personalizados para validar la capacidad del modelo. Ejemplo de ejecución:

```bash
# Ejecución del script principal
python main.py
```

El script evaluará el modelo y mostrará predicciones de ejemplos predefinidos.

## Resumen de etapas o pasos  🔩

1. **Descarga de recursos:**
   - Descarga de stopwords y lematizadores necesarios de NLTK.
2. **Carga de datos:**
   - Importación del archivo CSV con los mensajes y clases.
3. **Preprocesamiento de datos:**
   - Limpieza, tokenización y lematización de los mensajes.
4. **Vectorización:**
   - Conversión de los mensajes en representaciones numéricas usando TF-IDF.
5. **Definición del modelo:**
   - Configuración de una red neuronal densa para clasificación binaria.
6. **Entrenamiento y validación:**
   - Entrenamiento del modelo con el conjunto de datos de entrenamiento.
7. **Evaluación:**
   - Medición del rendimiento del modelo con el conjunto de prueba.
8. **Predicciones personalizadas:**
   - Clasificación de mensajes nuevos proporcionados por el usuario.

## Gráficas de Entrenamiento  🔬

El proyecto incluye gráficas de precisión y pérdida para evaluar el rendimiento del modelo:

- **Precisión de entrenamiento y validación.**
- **Pérdida de entrenamiento y validación.**

Las gráficas se generan automáticamente y se visualizan al finalizar el entrenamiento.

## Estructura del Proyecto 🌐

```plaintext
.
├── data
│   └── Spam-Classification.csv
├── deps
│   └── requeriments.txt
├── spam_classifier.py
├── requirements.txt
└── README.md
```



