# Spam Classifier with Deep Learning

Este proyecto implementa un clasificador de mensajes spam utilizando aprendizaje profundo (deep learning). A través de la representación TF-IDF de los mensajes y una red neuronal densa, el modelo es capaz de distinguir entre mensajes spam y no spam.

## Descripción

El proyecto utiliza varias etapas para preprocesar los datos y construir un modelo de clasificación:

- **Preprocesamiento de texto:** Eliminación de stopwords, lematización y tokenización utilizando NLTK.
- **Representación de datos:** Conversión de los mensajes en vectores TF-IDF.
- **Construcción del modelo:** Creación de una red neuronal con TensorFlow y Keras para clasificar los mensajes.
- **Entrenamiento y evaluación:** Uso de datos de entrenamiento y prueba para entrenar y evaluar el rendimiento del modelo.
- **Predicciones personalizadas:** Clasificación de nuevos mensajes ingresados por el usuario.

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

