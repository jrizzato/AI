# Clasificación de Especies de Iris con Red Neuronal

## Descripción del Proyecto
Proyecto de deep learning que implementa una red neuronal para clasificar especies de flores Iris utilizando TensorFlow y Keras. El modelo predice la especie de Iris basándose en cuatro medidas de características: longitud del sépalo, ancho del sépalo, longitud del pétalo y ancho del pétalo.

## Conjunto de Datos
- Fuente: Conjunto de datos de la flor Iris
- Características: 
  - Longitud del Sépalo
  - Ancho del Sépalo
  - Longitud del Pétalo
  - Ancho del Pétalo
- Variable Objetivo: Especies de Iris (3 clases) Sépalo, Versicolor o Virginica

## Requisitos Previos
- Python 3.8+
- Bibliotecas:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - keras
  - matplotlib

## Pasos de Preprocesamiento
1. Cargar datos desde CSV
2. Codificar etiquetas de especies numéricamente
3. Escalar características usando StandardScaler
4. Codificar variable objetivo con one-hot encoding
5. Dividir datos en conjuntos de entrenamiento y prueba

## Arquitectura del Modelo
- Capa de Entrada: 4 características
- Capa Oculta 1: 128 neuronas (activación ReLU)
- Capa Oculta 2: 128 neuronas (activación ReLU)
- Capa de Salida: 3 neuronas (activación Softmax)

### Parámetros de Entrenamiento
- Función de Pérdida: Entropía Cruzada Categórica
- Optimizador: Adam
- Tamaño de Lote: 16
- Épocas: 10
- Proporción de Validación: 20%

## Resultados
El modelo logra alta precisión en la clasificación de especies de Iris basándose en medidas de flores.

## Visualización
La precisión del entrenamiento se visualiza usando matplotlib, mostrando mejoras de precisión a través de las épocas.
