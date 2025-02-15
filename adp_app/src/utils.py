import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def format_currency(value):
    """Formate les valeurs monétaires"""
    return f"${value:,.2f}"

def create_kpi_metrics(df):
    """Crée les métriques KPI pour le dashboard"""
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100
    
    return {
        'Total Sales': format_currency(total_sales),
        'Total Profit': format_currency(total_profit),
        'Profit Margin': f"{profit_margin:.1f}%"
    }

def create_confusion_matrix_plot(conf_matrix, classes):
    """Crée une visualisation de la matrice de confusion"""
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=classes,
        y=classes,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis_title='Prédictions',
        yaxis_title='Valeurs Réelles',
        template="plotly_white"
    )
    
    return fig

def download_predictions(predictions_df):
    """Prépare le téléchargement des prédictions"""
    return predictions_df.to_csv(index=False).encode('utf-8')