import pytest
import pandas as pd
from src.model_prediction import predict

def test_predict():
    # Cargar un conjunto de datos de prueba
    test_data = pd.DataFrame({
        'Temperature (K)': [3000, 5000, 7000],
        'Luminosity (L/Lo)': [0.001, 1.0, 100.0],
        'Radius (R/Ro)': [0.1, 1.0, 10.0],
        'Absolute magnitude (Mv)': [15, 5, -5]
    })
    
    # Realizar predicciones
    predictions = predict(test_data)
    
    # Verificar que las predicciones no sean nulas
    assert predictions is not None
    assert len(predictions) == len(test_data)
    
    # Verificar que las predicciones sean de tipo esperado (por ejemplo, categorías de estrellas)
    expected_classes = ['Brown Dwarf', 'Main Sequence', 'Supergiant']  # Ajustar según el modelo
    assert all(pred in expected_classes for pred in predictions)