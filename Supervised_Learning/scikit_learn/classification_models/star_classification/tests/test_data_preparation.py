import pandas as pd
import pytest
from src.data_preparation import load_data, clean_data, encode_categorical_features

def test_load_data():
    df = load_data('data/star_classification.csv')
    assert isinstance(df, pd.DataFrame), "The loaded data should be a DataFrame"
    assert not df.empty, "The DataFrame should not be empty"

def test_clean_data():
    df = load_data('data/star_classification.csv')
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0, "There should be no missing values after cleaning"

def test_encode_categorical_features():
    df = load_data('data/star_classification.csv')
    cleaned_df = clean_data(df)
    encoded_df = encode_categorical_features(cleaned_df)
    assert 'Star category' not in encoded_df.columns, "The 'Star category' column should be encoded"
    assert 'Star color' not in encoded_df.columns, "The 'Star color' column should be encoded"
    assert 'Star type' in encoded_df.columns, "The 'Star type' column should remain in the DataFrame"