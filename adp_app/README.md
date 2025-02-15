Projet d'Analyse et Modélisation de Données avec Machine Learning


Auteur : Emac SAH

Description

Ce projet consiste en une analyse approfondie d'un ensemble de données de ventes, suivie de la conception d'une application web interactive à l'aide de Streamlit. L'objectif est d'extraire des insights stratégiques et exploitables à partir des données et de les visualiser de manière interactive.
Objectifs

  - Analyse Exploratoire des Données (EDA) : Comprendre la structure des données, identifier les valeurs manquantes, anomalies et doublons, et effectuer une analyse descriptive.
  - Nettoyage et Préparation des Données : Gestion des valeurs manquantes, standardisation des formats, et transformation des variables.
  - Modélisation : Création et évaluation de différents modèles de machine learning pour la classification et la régression.
  - Développement d'une Application Web : Conception d'une interface utilisateur interactive pour la visualisation des résultats.

Outils et Technologies

  - Bibliothèques Python : Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn.
  - Framework Web : Streamlit.
  - Modèles de Machine Learning : Régression Linéaire, Régression Logistique, Random Forest, KNN, Arbre de Décision, SVM.

Résultats et Insights

  - Identification des relations entre variables et des tendances saisonnières.
  - Analyse des performances financières et des segments de marché.
  - Comparaison des performances des différents modèles de machine learning.

Conclusion

Le projet met en lumière l'importance de l'analyse de données et du machine learning dans la prise de décision stratégique. L'application web développée permet une exploration dynamique des données, facilitant ainsi l'interprétation des résultats et la prise de décision.


## Installation

```bash
pip install -r requirements.txt

## Utilisation

streamlit run app.py

## Structure du projet

adp_app/
├── .streamlit/
│   └── config.toml
├── data/
│   └── Superstore2023.xlsx
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── visualization.py
│   ├── modeling.py
│   └── utils.py
├── requirements.txt
├── setup.py
├── README.md
└── app.py