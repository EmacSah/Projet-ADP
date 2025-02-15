from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

class SuperstoreModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.features = [
            'Ship_Mode', 'City', 'State',
            'Quantity', 'Discount', 'Profit'
        ]
    
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Effectue des prédictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Retourne l'importance des variables"""
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance
    
    def evaluate(self, X_test, y_test):
        """Évalue les performances du modèle"""
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': pd.DataFrame(report).transpose(),
            'confusion_matrix': conf_matrix
        }