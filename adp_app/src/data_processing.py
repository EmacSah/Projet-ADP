import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    """Charge et prépare les données initiales"""
    try:
        # Utilisation de Path pour gérer les chemins de fichiers
        file_path = Path(__file__).parent.parent / 'data' / 'Superstore2023.xlsx'
        df = pd.read_excel(str(file_path), sheet_name="Superstore2023")
        return df
    except Exception as e:
        raise Exception(f"Erreur lors du chargement des données: {e}")

def clean_data(df):
    """Nettoie et prépare les données"""
    df_clean = df.copy()
    
    # Traitement des valeurs manquantes
    df_clean.dropna(subset=['Sales', 'Quantity', 'Discount', 'Profit'], inplace=True)
    
    # Conversion des types
    if 'Postal Code' in df_clean.columns:
        df_clean['Postal Code'] = df_clean['Postal Code'].astype(str)
    
    return df_clean

def prepare_model_data(df, features, target='Segment'):
    """Prépare les données pour la modélisation"""
    encoder = LabelEncoder()
    scaler = StandardScaler()
    
    df_model = df.copy()
    
    # Encodage des variables catégoriques
    for col in features:
        if df_model[col].dtype == 'object':
            df_model[col] = encoder.fit_transform(df_model[col])
    
    # Normalisation des variables numériques
    numeric_features = df_model[features].select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_features) > 0:
        df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])
    
    X = df_model[features]
    y = df_model[target]
    
    return X, y, encoder, scaler