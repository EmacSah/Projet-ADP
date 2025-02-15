import subprocess
import sys
import pkg_resources
import streamlit as st

def check_package_version(package_name, required_version):
    """
    Vérifie si un package est installé avec la version requise
    Retourne True si le package est correctement installé, False sinon
    """
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        return installed_version == required_version
    except pkg_resources.DistributionNotFound:
        return False

def verify_dependencies():
    """
    Vérifie toutes les dépendances requises
    Retourne True si toutes les dépendances sont correctement installées
    """
    dependencies = {
        'streamlit': '1.31.1',
        'pandas': '2.1.4',
        'numpy': '1.24.3',
        'plotly': '5.18.0',
        'scikit-learn': '1.3.2',
        'seaborn': '0.13.0',
        'openpyxl': '3.1.2',
        'scipy': '1.11.3',
        'matplotlib': '3.7.1'
    }

    missing_packages = []
    incorrect_versions = []

    for package, version in dependencies.items():
        if not check_package_version(package, version):
            try:
                current_version = pkg_resources.get_distribution(package).version
                incorrect_versions.append(f"{package} (requis: {version}, installé: {current_version})")
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)

    if missing_packages or incorrect_versions:
        st.error("Problèmes de dépendances détectés !")
        
        if missing_packages:
            st.warning(f"Packages manquants : {', '.join(missing_packages)}")
        
        if incorrect_versions:
            st.warning(f"Versions incorrectes : {', '.join(incorrect_versions)}")
        
        try:
            st.info("Installation des dépendances manquantes...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            st.success("Dépendances installées avec succès !")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors de l'installation des dépendances : {str(e)}")
            return False
    
    return True

def import_dependencies():
    """
    Importe les dépendances après vérification
    Retourne un dictionnaire des modules importés
    """
    if not verify_dependencies():
        st.error("Impossible de continuer sans les dépendances requises")
        st.stop()

    try:
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import seaborn as sns
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt

        return {
            'pd': pd,
            'np': np,
            'px': px,
            'sns': sns,
            'LabelEncoder': LabelEncoder,
            'StandardScaler': StandardScaler,
            'train_test_split': train_test_split,
            'plt': plt
        }
    except ImportError as e:
        st.error(f"Erreur lors de l'importation des modules : {str(e)}")
        st.stop()