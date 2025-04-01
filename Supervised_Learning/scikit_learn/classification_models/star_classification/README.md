# Supervised Learning Project

Este proyecto tiene como objetivo desarrollar un modelo de clasificación multiclase utilizando el conjunto de datos de clasificación de estrellas. El modelo se entrenará para predecir diferentes tipos de estrellas basándose en características como temperatura, luminosidad, radio y magnitud absoluta.

## Estructura del Proyecto

- **data/**: Contiene el conjunto de datos y la documentación relacionada.
  - **star_classification.csv**: Conjunto de datos utilizado para el modelo de clasificación.
  - **README.md**: Información sobre el conjunto de datos, incluyendo su origen y descripción de las columnas.

- **notebooks/**: Incluye cuadernos de Jupyter para la exploración de datos.
  - **data_exploration.ipynb**: Análisis descriptivo y visualizaciones del conjunto de datos.

- **src/**: Contiene los scripts para la preparación de datos, entrenamiento del modelo, evaluación y predicción.
  - **data_preparation.py**: Funciones para cargar y preparar los datos.
  - **model_training.py**: Lógica para crear y entrenar el modelo de clasificación.
  - **model_evaluation.py**: Funciones para evaluar el rendimiento del modelo.
  - **model_prediction.py**: Funciones para realizar predicciones con el modelo entrenado.
  - **utils.py**: Funciones auxiliares para visualización y configuración.

- **tests/**: Contiene pruebas unitarias para asegurar la calidad del código.
  - **test_data_preparation.py**: Pruebas para las funciones de preparación de datos.
  - **test_model_training.py**: Pruebas para las funciones de entrenamiento del modelo.
  - **test_model_evaluation.py**: Pruebas para las funciones de evaluación del modelo.
  - **test_model_prediction.py**: Pruebas para las funciones de predicción.

- **requirements.txt**: Lista de dependencias necesarias para el proyecto.

## Instalación

Para instalar las dependencias del proyecto, ejecute el siguiente comando:

```
pip install -r requirements.txt
```

## Uso

1. Realice la exploración de datos utilizando el cuaderno `data_exploration.ipynb`.
2. Prepare los datos ejecutando el script `data_preparation.py`.
3. Entrene el modelo utilizando `model_training.py`.
4. Evalúe el modelo con `model_evaluation.py`.
5. Realice predicciones con `model_prediction.py`.

## Contribuciones

Las contribuciones son bienvenidas. Si desea contribuir, por favor abra un issue o un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.