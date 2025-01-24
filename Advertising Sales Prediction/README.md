https://github.com/LinkedInLearning/artificial-intelligence-foundations-neural-networks-4381282

# Advertising Sales Prediction

Este repositorio contiene un script en Python que utiliza aprendizaje automático para predecir las ventas (“sales”) basándose en diferentes canales de publicidad (digital, TV, radio y periódico).

## Funcionalidades del Script

1. **Carga y exploración de datos**

   - Lee un archivo CSV llamado `Advertising_2023.csv` con información sobre inversiones publicitarias y ventas.
   - Realiza una exploración inicial de los datos:
     - Muestra las primeras filas del conjunto de datos.
     - Proporciona estadísticas descriptivas y detalles sobre la estructura del DataFrame.

2. **Preparación de los datos**

   - Separa las columnas de características (`digital`, `TV`, `radio`, `newspaper`) y las etiquetas (`sales`).
   - Normaliza las características para garantizar una escala uniforme, mejorando la eficiencia del modelo.
   - Divide los datos en conjuntos de entrenamiento (60%) y prueba (40%) para evaluar el modelo de manera objetiva.

3. **Diseño y entrenamiento del modelo**

   - Construye un modelo de red neuronal utilizando Keras con las siguientes capas:
     - Capa densa con 4 neuronas y función de activación ReLU.
     - Capa densa con 3 neuronas y función de activación ReLU.
     - Capa de salida con 1 neurona.
   - Compila el modelo utilizando:
     - Optimizador: `adam`
     - Función de pérdida: `mse` (error cuadrático medio).
     - Métrica: `mse`.
   - Entrena el modelo durante 32 épocas, validando los resultados con el conjunto de prueba.

4. **Evaluación y visualización del rendimiento**

   - Genera gráficas que muestran la pérdida (“loss”) durante el entrenamiento y la validación.
   - Realiza predicciones en el conjunto de prueba y las compara con los valores reales utilizando:
     - Gráficas de dispersión entre valores reales y predicciones.
     - Cálculo del error cuadrático medio (RMSE).

5. **Exportación de resultados**

   - Crea un DataFrame que combina los valores reales y las predicciones para facilitar el análisis posterior.

## Requisitos

Para ejecutar el script, necesitas instalar las siguientes bibliotecas:

- `pandas`
- `keras`
- `scikit-learn`
- `matplotlib`
- `numpy`

Puedes instalarlas ejecutando:

```bash
pip install pandas keras scikit-learn matplotlib numpy
```

## Ejecución

1. Coloca el archivo `Advertising_2023.csv` en una carpeta llamada `data` dentro del directorio donde se encuentra el script.
2. Ejecuta el script con:

```bash
python script.py
```

## Resultados Esperados

- Una visualización del progreso de la pérdida durante el entrenamiento y la validación.
- Una gráfica de dispersión mostrando la relación entre las predicciones y los valores reales.
- Errores cuadráticos medios (RMSE) para los conjuntos de entrenamiento y prueba.

## Nota

El script incluye comentarios detallados que explican cada paso del proceso. Si deseas explorar más detalles, puedes descomentar las líneas relacionadas con la inspección de datos o visualizaciones adicionales.

---
