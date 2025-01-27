# Análisis de Causas Raíz de Problemas en IT

## Descripción del Proyecto
Este proyecto utiliza un modelo de redes neuronales para identificar la causa raíz de problemas IT a partir de un conjunto de datos de características relacionadas con el rendimiento del sistema. Las causas raíz incluyen problemas de base de datos, fugas de memoria y retrasos en la red.

## Conjunto de Datos
- **Fuente:** Archivo `root_cause_analysis.csv`.
- **Características:**
  - `CPU_LOAD`
  - `MEMORY_LEAK_LOAD`
  - `DELAY`
  - `ERROR_1000`, `ERROR_1001`, `ERROR_1002`, `ERROR_1003`
- **Variable Objetivo:**
  - `ROOT_CAUSE`, con tres posibles valores: `DATABASE_ISSUE`, `MEMORY_LEAK`, `NETWORK_DELAY`.

## Requisitos Previos
- Python 3.8+
- Bibliotecas necesarias:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`
  - `scikit-learn`

Instala las dependencias utilizando:
```bash
pip install -r deps/requirements.txt
```

## Pasos de Preprocesamiento
1. **Carga de datos:** Se cargan los datos desde el archivo CSV.
2. **Codificación:** Las etiquetas de las causas raíz (`ROOT_CAUSE`) se codifican a valores numéricos utilizando `LabelEncoder`.
3. **División de datos:** Los datos se dividen en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba).
4. **Normalización:** Las características se escalan para mejorar el rendimiento del modelo.

## Arquitectura del Modelo
El modelo es una red neuronal creada con Keras, que incluye:
- Entrada: 7 nodos para las características.
- Capas ocultas:
  - 3 capas densas con 32 neuronas y activación `relu`.
  - Capas `Dropout` para prevenir el sobreajuste (0.25).
- Capa de salida: 3 nodos con activación `softmax` para clasificación multiclase.

### Parámetros de Entrenamiento
- **Función de pérdida:** `sparse_categorical_crossentropy`
- **Optimizador:** `adam`
- **Métrica:** Precisión (`accuracy`)
- **Épocas:** 50
- **Tamaño del lote:** 10

## Resultados
- Precisión en los datos de prueba: ~90% (dependiendo de la ejecución).
- Reporte de clasificación:
  - Incluye precisión, exhaustividad y puntaje F1 para cada clase.
- Matriz de confusión para visualizar errores de clasificación.

## Visualización
- **Gráficos del entrenamiento:**
  - Evolución de la precisión y la pérdida durante las épocas.
- **Matriz de confusión:**
  - Representación visual del desempeño del modelo en los datos de prueba.

![Gráfico de Precisión y Pérdida](ruta/a/grafico_accuracy_loss.png)
![Matriz de Confusión](ruta/a/matriz_confusion.png)

---
Si tienes preguntas o sugerencias, por favor crea un issue o envía un pull request. ¡Gracias por contribuir!

