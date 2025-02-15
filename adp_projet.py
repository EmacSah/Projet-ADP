#!/usr/bin/env python
# coding: utf-8

# # -*- coding: utf-8 -*-
# """Projet_MLP_M2DA3.ipynb
# 
# Original file is located at
#     https://colab.research.google.com/drive/1OpkmSFElJFmYHDxIYLZcQlneZpUcL5iV
# 
# # **MODULE MACHINE LEARNING AVEC PYTHON**
# ---------------------------------------------------------------------------
# RAPPORT REDIGE ET PRESENTE PAR :
# 
# *   Emac SAH
# 
# Etudianst en M2 DA3 - iA SCHOOL
# 
# ---------------------------------------------------------------------------
# 
# **Présentation Structurée du Projet : Machine Learning avec Python**
# 
# 1. Contexte et Objectif
# 
# Notre projet de machine learning avec python a pour objectif de tirer des informations stratégiques et exploitables à partir des données fournies de notre dataset et d'écrire les différents modèles de regression et classification. Selon le temps imparti,concevoir une application web interactive (avec un framework comme Streamlit ou Dash/Plotly) qui permettra de visualiser et d’explorer les résultats de cette analyse.
# 
# 2. Démarche et Approche
# 
# La démarche se déroulera en plusieurs étapes méthodiques :
# 
#     **Exploration des Données :**
#         Compréhension de la structure des données (colonnes, types de données, dimensions).
#         Identification des valeurs manquantes, anomalies et doublons.
#         Analyse descriptive pour avoir une vue d’ensemble des tendances et distributions.
# 
#     **Nettoyage et Préparation des Données :**
#         Gestion des valeurs manquantes et aberrantes.
#         Standardisation des formats et transformation des variables selon les besoins des analyses.
# 
#     **Analyse Exploratoire des Données (EDA) :**
#         Identifier des relations entre variables.
#         Comprendre les comportements à travers des visualisations claires (boxplots, heatmaps, scatterplots, etc.).
# 
#     **Analyses Avancées :**
#         Prédictions financières, saisonnières, ou géographiques si pertinentes.
#         Analyse des clusters ou segments pour catégoriser les données.
#         Identifier les KPI clés pour des tableaux de bord interactifs.
# 
#     **Machine Learning :**
#         Création et des différents models de classification et regression.
#         Intéprétation des résultats et recommandations.
# 
# 
#     **Développement de l'Application Web :**
#         Conception de l’interface utilisateur pour la visualisation des résultats.
#         Intégration des modèles et graphiques dans une application web pour l’exploration dynamique.
# 
# # **Outils et Technologies**
# 
# Bibliothèques Python :
# 
# - Pandas, NumPy pour l’analyse de données.
# - Matplotlib, Seaborn, Plotly pour la visualisation.
# - Scikit-learn pour les modèles prédictifs.
# 
# Framework Web :
# 
# - Streamlit ou Dash pour l’application interactive.
# 
# # **Plan de Travail**
# 
# # **Étape 1 : Analyse Préliminaire des Données**
# 
# **Tâches :**
# 
#   - Charger les données et examiner leur structure.
#   - Identifier les problèmes de qualité des données.
#   - Réaliser une description statistique initiale (moyennes, médianes, écart-types).
# 
# **Livrables :**
# 
#   - Rapport d’audit des données (problèmes identifiés, suggestions de nettoyage).
# """

# In[11]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install millify')


# In[14]:


#Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
import plotly.figure_factory as ff
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from millify import millify
import os
import base64


# In[17]:


# Définir le chemin des fichiers (logo et données)
base_path = os.getcwd()
excel_path = os.path.join(base_path, 'Superstore2023.xlsx')

# Charger les données
df = pd.read_excel(excel_path, sheet_name="Superstore2023")


# In[18]:


#Affichage des 5 premières lignes du fichier avec résumé du nombre de colonnes
print(df.head(5))

# Résumé des colonnes
print(df.info())

# Résumé statistique
print(df.describe())

#Vérification des valeurs manquantes
print(df.isnull().sum())


# """**Résultats de l'Analyse Préliminaire**
# 
# Structure des Données
# 
#  - Le fichier contient 1 feuille : Superstore2023.
# 
#  - Cette feuille contient 21 colonnes et 9,994 lignes.
# 
# ---------------------------------------------------------------------------
# # **Résumé des Colonnes**
# 
#   **Dates :**
# 
#   Colonnes : Order Date, Ship Date (format datetime64).Utiles pour des analyses temporelles (ex. délais d'expédition).
# 
# 
#   **Numériques :**
# 
#   Colonnes : Sales, Quantity, Discount, Profit, Postal Code.Ces colonnes permettront d'analyser les performances financières et les distributions géographiques.
# 
# 
#   **Catégoriques :**
#   
#   Exemples : Ship Mode, Segment, Region, Category, Sub-Category. Serons utiliser pour la segmentation et la visualisation.
# 
# -------------------------------------------------------------------------
# # **Statistiques Descriptives**
# 
# **Ventes (Sales) :**
#   - Moyenne : 100,868.7
#   - Écart type : 518,426.7 (indique une forte variabilité).
#   - Maximum : 23,962,660 (indique des transactions potentiellement extrêmes).
# 
# **Profit :**
#   - Moyenne : 356,809,900.
#   - Minimum : -509,997,000,000 (grandes pertes à examiner).
#   - Maximum : 99,432,000,000.
#   
# -------------------------------------------------------------------------
# #  **Problèmes Identifiés :**
#        
# **Valeurs manquantes :**
#   - Postal Code : 11 valeurs manquantes.
#   - Sales, Quantity, Discount, Profit : 6 valeurs manquantes.
#         
# **Anomalies potentielles :**
#   - Profit a des valeurs négatives (pertes importantes).
#   - Sales présente des valeurs extrêmes (pouvant être des outliers).
#         
# **Colonnes à vérifier :**
#   - Les types float pour Postal Code et d'autres colonnes méritent une vérification.
# """

# In[19]:


# Visualisation des distributions
import matplotlib.pyplot as plt
df['Sales'].hist(bins=50)
plt.show()


# """# **Étape 2 : Nettoyage et Préparation des Données**
# 
# **Tâches :**
# 
#   - Traiter les valeurs manquantes (remplacement, suppression ou imputation).
#   - Convertir les types de données et gérer les formats .
#   - Ajouter des variables dérivées si nécessaire
# 
# **Livrables :**
# 
#   - Base de données propre prête pour l’analyse.
# 
# **Gestion des Valeurs Manquantes**
# 
# - Postal Code : Imputation possible (utilisation du mode ou une correspondance basée sur City et State).
# - Colonnes Numériques : Imputation par la médiane ou la moyenne ou suppression des lignes avec des valeurs manquantes si elles ne sont pas nombreuses
# """

# In[20]:


#duplication du dataframe
df_clean = df.copy()

# Vérification et suppression des doublons
initial_rows = df_clean.shape[0]
df_clean.drop_duplicates(inplace=True)
duplicates_removed = initial_rows - df_clean.shape[0]
print(f"Nombre de doublons supprimés : {duplicates_removed}")

# Traitement des valeurs manquantes
missing_values = df_clean.isnull().sum()

# Imputation pour 'Postal Code' par le mode (remplissage par la valeur la plus fréquente)
if 'Postal Code' in missing_values and missing_values['Postal Code'] > 0:
    postal_mode = df_clean['Postal Code'].mode()[0]
    df_clean['Postal Code'].fillna(postal_mode)

# Suppression des lignes avec des valeurs critiques manquantes
df_clean.dropna(subset=['Sales', 'Quantity', 'Discount', 'Profit'], inplace=True)

# Vérification des valeurs manquantes après traitement
missing_after_cleaning = df_clean.isnull().sum()
print("Valeurs manquantes après nettoyage :")
print(missing_after_cleaning)

# Conversion du type 'Postal Code' en chaîne (string)
df_clean['Postal Code'] = df_clean['Postal Code'].astype(str)

# Identification des outliers à l'aide du Z-Score pour 'Sales' et 'Profit'
from scipy.stats import zscore

df_clean['Sales_Z'] = zscore(df_clean['Sales'])
df_clean['Profit_Z'] = zscore(df_clean['Profit'])

# Filtrer les outliers
outliers_sales = df_clean[(df_clean['Sales_Z'] > 3) | (df_clean['Sales_Z'] < -3)]
outliers_profit = df_clean[(df_clean['Profit_Z'] > 3) | (df_clean['Profit_Z'] < -3)]

print(f"Nombre d'outliers pour 'Sales' : {outliers_sales.shape[0]}")
print(f"Nombre d'outliers pour 'Profit' : {outliers_profit.shape[0]}")

# Nettoyer les colonnes temporaires de Z-Score si vous ne souhaitez pas les conserver
df_clean.drop(columns=['Sales_Z', 'Profit_Z'], inplace=True)

# Résumé final
print(f"Nombre total de lignes après nettoyage : {df_clean.shape[0]}")


# """# **Résultats du Nettoyage des Données**
# 
# **1. Doublons**
#    
#    Aucun doublon trouvé et supprimé.
# 
# 
# **2. Valeurs Manquantes**
# 
#    Colonnes concernées avant nettoyage : Postal Code (11 valeurs), Sales, Quantity, Discount, Profit (6 valeurs chacune).
# 
#  - Actions effectuées :
# 
#      * Imputation de Postal Code avec la valeur la plus fréquente.
# 
#      * Suppression des lignes avec des valeurs manquantes critiques.
# 
#      * Valeurs manquantes après nettoyage : Aucune.
# 
# **3. Anomalies (Outliers)**
# 
#   - Outliers identifiés avec un Z-Score > 3 ou < -3 :
# 
#      * Ventes (Sales) : 94 lignes.
# 
#      * Profit (Profit) : 102 lignes.
# 
# Les outliers n'ont pas encore été supprimés pour permettre une analyse approfondie et déterminer leur importance.
# 
# 
# **4. Données Après Nettoyage**
# 
#  - Lignes restantes : 9,988 (après suppression des valeurs manquantes).
#  - Colonnes : 21.
#  - Les types de données sont désormais corrects (Postal Code est une chaîne, les dates sont au format datetime).
# 
#  --------------------------------------------------------------------------
# 
# # **Étape 3 : Analyse Exploratoire des Données (EDA)**
# 
# Tâches :
# 
#   - Visualiser les données pour identifier des tendances et anomalies.
#   - Calculer des corrélations et générer des graphiques pertinents.
#   - Effectuer des segmentations préliminaires.
# 
# Livrables :
# 
#   - Graphiques et tableaux synthétiques (corrélations, distributions).
#   - Insights clés documentés.
# 
# --------------------------------------------------------------------------
# Visualisations
# 
# 
#   - Distributions des variables : Histogrammes, boxplots.
#   - Corrélations : Matrice et heatmap.
#   - **Analyse** catégorielle : Répartition par segments, régions, etc.
# """

# In[21]:


# Aperçu rapide des données catégorielles
print(df_clean['Region'].value_counts())
print(df_clean['Category'].value_counts())


# Analyse Temporelle
# - Étudier les tendances des ventes et profits sur les dates (Order Date, Ship Date).
# - Calculer les délais d’expédition (Ship Date - Order Date).
# 

# In[22]:


# Calcul des délais d'expédition
df_clean['Shipping Delay'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days


# """Identification des Relations
# - Explorer les relations entre les variables (ex. Sales et Profit).
# """

# In[23]:


# Scatterplot pour étudier la relation entre Sales et Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Sales', y='Profit', data=df_clean)
plt.title('Relation entre Ventes (Sales) et Profits')
plt.show()


# # **Analyse Exploratoire avec PLOTLY**
# 
# #bibliothèques utilisées , elles sont importées depuis le debut
# #import numpy as np
# #import plotly.figure_factory as ff
# #import plotly.express as px # Import plotly.express

# In[24]:


# Calculer la matrice de corrélation
corr_matrix = df_clean[['Sales', 'Profit', 'Discount', 'Quantity']].corr().values
columns = ['Sales', 'Profit', 'Discount', 'Quantity']

# Heatmap
fig = ff.create_annotated_heatmap(
    z=corr_matrix,
    x=columns,
    y=columns,
    colorscale="Viridis",
    showscale=True
)
fig.update_layout(title="Matrice de Corrélation", template="plotly_white")
fig.show()


# # **ANALYSE DE LA MATRICE DE CORRELATION**
# 
# **Row_ID :** Row_ID ne présente pas de corrélation significative avec aucune autre variable. Cela indique qu’il n’a pas d’influence particulière sur les ventes, les quantités, les profits ou les retards.
# Sales (Ventes) :
# 
# **Quantité (0.0936) :** Corrélation positive faible, indiquant que des quantités plus élevées peuvent légèrement augmenter les ventes.
# 
# **Discount (0.106) :** Corrélation positive faible, suggérant que les remises sont légèrement associées à une augmentation des ventes.
# 
# 
# **Profit (-0.0033) :** Corrélation très faible et négative, montrant qu'il n'y a pratiquement aucune relation entre les ventes et le profit.
# 
# **Shipping_Delay (-0.0041) :** Corrélation très faible et négative, indiquant que les délais d'expédition n'influencent pas directement les ventes.
# 
# **Discount (0.0085) :**Corrélation très faible, montrant que les remises n'affectent pas significativement les quantités commandées.
# 
# **Profit (0.0210) :** Corrélation faible, suggérant qu’une augmentation des quantités peut être légèrement liée à une augmentation du profit.
# 
# **Shipping_Delay (0.0152) :** Corrélation faible, montrant que les quantités commandées ont un lien limité avec les retards de livraison.
# 
# 
# **Profit (-0.0161) :**  Corrélation négative très faible, indiquant que des remises plus importantes pourraient légèrement diminuer les profits.
# Shipping_Delay (0.0012) : Corrélation négligeable, montrant que les remises n’ont presque aucun impact sur les retards.
# Profit :
# 
# **Shipping_Delay (-0.0105) :** Corrélation négative très faible, indiquant que les retards n’ont pas d’influence significative sur les profits.
# Shipping_Delay (Retards de livraison) :
# 
# Toutes les corrélations sont très faibles, ce qui signifie que les retards d'expédition n'ont pas de relation forte avec les autres variables.
# 
# # **Conclusion**
# 
# La matrice nous montre que la plupart des relations entre les variables sont faibles, voire insignifiantes.
# Les variables ayant les corrélations légèrement plus élevées sont :
# - Sales et Discount (0.106) : Les remises semblent avoir un léger impact positif sur les ventes.
# - Sales et Quantity (0.0936) : Les ventes augmentent légèrement avec les quantités.
# - Quantity et Profit (0.021) : Une relation faible mais positive.
# 

# In[25]:


# Distribution des délais d'expédition
fig = px.histogram(df_clean, x='Shipping Delay', nbins=20, title="Distribution des Délais d'Expédition")
fig.update_traces(marker_color="green")
fig.update_layout(xaxis_title="Shipping Delay (days)", yaxis_title="Frequency", template="plotly_white")
fig.show()

# Créer une colonne 'Order Month' comme période
df_clean['Order Month'] = df_clean['Order Date'].dt.to_period('M')

# Groupement des ventes par 'Order Month'
sales_trend = df_clean.groupby('Order Month')['Sales'].sum().reset_index()

# Convertion 'Order Month' en string pour éviter les erreurs
sales_trend['Order Month'] = sales_trend['Order Month'].astype(str)

# Création du graphique avec Plotly
import plotly.express as px
fig = px.line(sales_trend, x='Order Month', y='Sales', title="Tendance des Ventes dans le Temps")
fig.update_layout(xaxis_title="Mois", yaxis_title="Ventes", template="plotly_white")
fig.show()

#Relation entre Ventes (Sales) et Profits (Profit)
fig = px.scatter(df_clean, x='Sales', y='Profit', title="Relation entre Ventes (Sales) et Profits", opacity=0.7)
fig.update_layout(xaxis_title="Sales", yaxis_title="Profit", template="plotly_white")
fig.show()


# **Total des Ventes par Région et Département**

# In[26]:


fig = px.bar(df_clean, x='Region', y='Sales', color='State', title="Total des Ventes par Région et Département")
fig.update_layout(xaxis_title="Région", yaxis_title="Ventes Totales", template="plotly_white")
fig.show()


# """**Total Profit par Région**"""

# In[27]:


fig = px.bar(df_clean, x='Region', y='Profit', title="Profit par Région", color='Region')
fig.update_layout(xaxis_title="Région", yaxis_title="Profit Total", template="plotly_white")
fig.show()


# """**Profit par Semaine**"""

# In[28]:


df_clean['Order Week'] = df_clean['Order Date'].dt.to_period('W').astype(str)
weekly_profit = df_clean.groupby('Order Week')['Profit'].sum().reset_index()

fig = px.line(weekly_profit, x='Order Week', y='Profit', title="Profit par Semaine")
fig.update_layout(xaxis_title="Semaine", yaxis_title="Profit", template="plotly_white")
fig.show()


# """**Ventes par Mode d’Expédition**"""

# In[29]:


fig = px.pie(df_clean, names='Ship Mode', values='Sales', title="Ventes par Mode d'Expédition")
fig.update_layout(template="plotly_white")
fig.show()


# """**Ventes par Région-Département et Produit**"""

# In[30]:


# Treemap initial sans filtre
fig = px.treemap(
    df_clean,
    path=['Region', 'State', 'Product Name'],  # Hiérarchie : Région → Département → Produit
    values='Sales',
    color='Region',  # Couleurs par Région
    title="Ventes par Région, Département et Produit",
    labels={'Region': 'Région', 'State': 'Département', 'Product Name': 'Produit'}
)

# Ajouter un menu dropdown pour filtrer par Région
fig.update_layout(
    updatemenus=[
        dict(
            buttons=[
                # Option pour afficher toutes les données
                dict(label="Toutes les Régions",
                     method="restyle",
                     args=[{"values": [df_clean['Sales']],
                            "ids": [df_clean['Region'] + " - " + df_clean['State'] + " - " + df_clean['Product Name']]}]),

                # Option pour afficher uniquement les données de chaque Région
                dict(label="West",
                     method="restyle",
                     args=[{"values": [df_clean[df_clean['Region'] == 'West']['Sales']],
                            "ids": [df_clean[df_clean['Region'] == 'West']['State'] + " - " + df_clean['Product Name']]}]),

                dict(label="East",
                     method="restyle",
                     args=[{"values": [df_clean[df_clean['Region'] == 'East']['Sales']],
                            "ids": [df_clean[df_clean['Region'] == 'East']['State'] + " - " + df_clean['Product Name']]}]),

                dict(label="Central",
                     method="restyle",
                     args=[{"values": [df_clean[df_clean['Region'] == 'Central']['Sales']],
                            "ids": [df_clean[df_clean['Region'] == 'Central']['State'] + " - " + df_clean['Product Name']]}]),

                dict(label="South",
                     method="restyle",
                     args=[{"values": [df_clean[df_clean['Region'] == 'South']['Sales']],
                            "ids": [df_clean[df_clean['Region'] == 'South']['State'] + " - " + df_clean['Product Name']]}]),
            ],
            direction="down",
            showactive=True,
            x=0.1,  # Position horizontale
            y=1.2,  # Position verticale
        ),
    ]
)

# Mettre à jour le style général
fig.update_layout(template="plotly_white")

# Afficher le graphique
fig.show()


# In[31]:


fig = px.sunburst(df_clean, path=['Region', 'State', 'Product Name'], values='Sales', title="Ventes par Région, Département et Produit")
fig.update_layout(template="plotly_white")
fig.show()


# In[32]:


# Créer le scatterplot avec des filtres interactifs
fig = px.scatter(
    df_clean,
    x='Region',
    y='State',
    size='Sales',  # Taille des points basée sur les ventes
    color='Category',  # Couleurs par catégorie
    hover_name='Product Name',
    facet_col='Category',  # Ajouter un filtre interactif par catégorie
    title="Scatterplot : Région-Département avec Filtres Interactifs",
    labels={'Region': 'Région', 'State': 'Département', 'Category': 'Catégorie'}
)

# Personnaliser l'affichage
fig.update_layout(
    xaxis_title="Région",
    yaxis_title="Département",
    template="plotly_white"
)

# Afficher le graphique
fig.show()


# In[33]:


# Analyse des remises
discount_analysis = df_clean.groupby('Discount').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).reset_index()

discount_analysis['Profit Margin (%)'] = (discount_analysis['Profit'] / discount_analysis['Sales']) * 100

# Afficher le tableau dans Colab
from IPython.display import display
display(discount_analysis)


# In[34]:


fig = px.scatter(
    df_clean,
    x='Discount',
    y='Profit',
    size='Sales',
    color='Region',
    title="Impact des Remises sur les Profits",
    labels={'Discount': 'Remises', 'Profit': 'Profits'}
)

fig.update_layout(template="plotly_white")
fig.show()


# In[35]:


fig = px.histogram(
    df_clean,
    x='Discount',
    title="Distribution des Remises",
    labels={'Discount': 'Remises'},
    nbins=20,
    color='Region'
)

fig.update_layout(template="plotly_white")
fig.show()


# In[36]:


discount_profit_avg = df_clean.groupby('Discount')['Profit'].mean().reset_index()


# In[37]:


fig = px.bar(
    discount_profit_avg,
    x='Discount',
    y='Profit',
    title="Profit Moyen par Taux de Remise",
    labels={'Discount': 'Remises', 'Profit': 'Profit Moyen'}
)

fig.update_layout(template="plotly_white")
fig.show()


# In[38]:


# Analyse par catégorie
category_analysis = df_clean.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).reset_index()

category_analysis['Profit Margin (%)'] = (category_analysis['Profit'] / category_analysis['Sales']) * 100
display(category_analysis)


# # **Insights Clés Documentés**
# 
# Voici les insights tirés des analyses précédentes :
# 
# - Impact des Remises (Discounts) :
#     * Les remises les plus élevées (>20%) réduisent considérablement les marges bénéficiaires.
#     * Des remises modérées (10-20%) maintiennent un bon équilibre entre volume de ventes et profitabilité.
# 
# - Analyse par Catégorie :
#     * **Technologie** est la catégorie avec le chiffre d'affaires et la marge bénéficiaire les plus élevés.
#     * **Office Supplies** génère beaucoup de volume mais des marges plus faibles.
#     * **Furniture** est la catégorie la moins rentable, notamment à cause des coûts élevés.
# 
# - Analyse par Région :
#     * **West** est la région la plus performante en termes de ventes et de profits.
#     * **South** génère le moins de ventes et de profits, ce qui pourrait être une opportunité d'amélioration.
# 
# - Délais d'Expédition :
#         La majorité des expéditions sont livrées dans les 1-5 jours, mais quelques retards significatifs (>10 jours) impactent probablement la satisfaction client.
# 
# # **Étape 4 : Analyses Spécifiques**
# 
# 
# **Plan des Analyses**
# 
# - Analyses Financières :
#   * Identifier les produits/services les plus performants.
#   * Analyser les marges bénéficiaires et tendances de revenus.
# 
# - Analyses Saisonnières :
#    * Identifier les pics d’activité.
#    * Prévoir les périodes creuses et proposer des recommandations.
# 
# - Analyses Géographiques :
#   * Étudier la performance par région.
#   * Identifier des zones prioritaires pour des actions futures.
# 
# **Livrables :**
# 
#   * Modèles prédictifs.
#   * Résultats interprétés avec clarté.
# 
# **1. Analyses Financières**
# 
# a. Produits les Plus Performants
# 
# Critères :
# 
#  * Ventes les plus élevées.
#  * Profits les plus élevés.
# 

# In[39]:


# Produits avec les ventes les plus élevées
top_products_sales = df_clean.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index().sort_values(by='Sales', ascending=False).head(10)


# In[40]:


# Produits avec les profits les plus élevés
top_products_profit = df_clean.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index().sort_values(by='Profit', ascending=False).head(10)


# In[41]:


# Afficher les résultats
from IPython.display import display
display(top_products_sales)
display(top_products_profit)


# In[42]:


# Graphe pour les produits avec les ventes les plus élevées
fig = px.bar(
    top_products_sales,
    x='Product Name',
    y='Sales',
    title="Top 10 Produits par Ventes",
    text='Sales',
    labels={'Sales': 'Ventes', 'Product Name': 'Produit'},
    color='Sales'
)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(xaxis_tickangle=45, template="plotly_white")
fig.show()


# In[43]:


#Marges Bénéficiaires et Revenus
# Calculer les marges bénéficiaires par catégorie
category_margins = df_clean.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
category_margins['Profit Margin (%)'] = (category_margins['Profit'] / category_margins['Sales']) * 100

display(category_margins)


# In[44]:


# Graphe pour les produits avec les profits les plus élevés
fig = px.bar(
    top_products_profit,
    x='Product Name',
    y='Profit',
    title="Top 10 Produits par Profits",
    text='Profit',
    labels={'Profit': 'Profits', 'Product Name': 'Produit'},
    color='Profit'
)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(xaxis_tickangle=45, template="plotly_white")
fig.show()


# **2. Analyses Saisonnières**
#    Identifier les Pics d’Activité

# In[45]:


# S'assurer que les dates sont au format datetime
df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])

# Créer une colonne pour le mois
df_clean['Order Month'] = df_clean['Order Date'].dt.to_period('M')

# Agréger les ventes par mois
monthly_sales = df_clean.groupby('Order Month')['Sales'].sum().reset_index()

# Ajouter une colonne avec une moyenne glissante
monthly_sales['Sales Rolling Avg'] = monthly_sales['Sales'].rolling(window=3).mean()

# Convertir 'Order Month' en string pour l'utiliser dans le graphique
monthly_sales['Order Month'] = monthly_sales['Order Month'].astype(str)


# In[46]:


# Tracer le graphique des ventes et de la moyenne glissante
fig = px.line(
    monthly_sales,
    x='Order Month',
    y=['Sales', 'Sales Rolling Avg'],
    title="Prévision avec Moyenne Glissante",
    labels={'Order Month': 'Mois', 'value': 'Ventes'},
    template="plotly_white"
)
fig.update_layout(xaxis_title="Mois", yaxis_title="Ventes", legend_title="Légende")
fig.show()


# In[47]:


# Ajouter une moyenne glissante (fenêtre de 3 mois)
monthly_sales['Sales Rolling Avg'] = monthly_sales['Sales'].rolling(window=3).mean()


# In[48]:


# Tracer la tendance avec la moyenne glissante
fig = px.line(
    monthly_sales,
    x='Order Month',
    y=['Sales', 'Sales Rolling Avg'],
    title="Prévision des Ventes avec Moyenne Glissante",
    labels={'Order Month': 'Mois', 'value': 'Ventes'},
    template="plotly_white"
)
fig.update_layout(xaxis_title="Mois", yaxis_title="Ventes", legend_title="Légende")
fig.show()


# # **Analyse du Graphique : Prévision avec Moyenne Glissante**
# 
# # Résumé des Observations :
# 
# #**Tendance Générale :**Le graphique montre une tendance globale à la hausse des ventes sur la période analysée (2020 à 2023), ce qui indique une croissance progressive des activités commerciales.
# 
# **Pics d'Activité :** 
#     Les pics les plus significatifs se situent autour de : 
#     - Fin 2020 (période de fin d'année).
#     - Début 2022 et mi-2023, qui pourraient correspondre à des événements commerciaux importants ou des campagnes promotionnelles.
# 
# **Périodes Creuses :**
# 
#     Des baisses significatives des ventes sont visibles :
# 
#     - Début 2021.
# 
#     - Début 2022.
# 
#     - Mi-2023, après les pics de ventes.
#     
#               
#     Moyenne Glissante (courbe rouge) :La moyenne glissante lisse les fluctuations pour révéler une tendance plus stable.
#     Elle confirme une progression régulière avec des variations modérées.

# **3. Analyses Géographiques**
# a. Performance par Région

# In[59]:


# Identifier les départements avec des profits négatifs
negative_profit_states = df_clean.groupby('State').agg({
    'Profit': 'sum'
}).reset_index()
negative_profit_states = negative_profit_states[negative_profit_states['Profit'] < 0]

display(negative_profit_states)

# Aggregate sales and profit by state and state code
state_sales_profit = df_clean.groupby('State').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

# Ajouter les abréviations des états
state_sales_profit['State Code'] = state_sales_profit['State']


# Carte choroplèthe des ventes
fig_sales = px.choropleth(
    state_sales_profit,
    locations='State Code',
    locationmode='USA-states',
    color='Sales',
    hover_name='State',
    hover_data={'Sales': True, 'Profit': False},  # Inclure uniquement les ventes
    color_continuous_scale='Viridis',
    title="Total des Ventes par État"
)

fig_sales.update_layout(
    geo_scope='usa',
    template="plotly_white"
)
fig_sales.show()


# In[60]:


# Carte choroplèthe des profits
fig_profit = px.choropleth(
    state_sales_profit,
    locations='State Code',
    locationmode='USA-states',
    color='Profit',
    hover_name='State',
    hover_data={'Sales': True, 'Profit': True},  # Afficher à la fois ventes et profits
    color_continuous_scale='Plasma',
    title="Total des Profits par État"
)

fig_profit.update_layout(
    geo_scope='usa',
    template="plotly_white"
)
fig_profit.show()


# In[61]:


# Marges par région
region_margins = df_clean.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
region_margins['Profit Margin (%)'] = (region_margins['Profit'] / region_margins['Sales']) * 100

# Graphique interactif
fig = px.bar(
    region_margins,
    x='Region',
    y='Profit Margin (%)',
    title="Marges Bénéficiaires par Région",
    text='Profit Margin (%)',
    labels={'Profit Margin (%)': 'Marge (%)', 'Region': 'Région'},
    color='Profit Margin (%)'
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(template="plotly_white")
fig.show()

# Agréger les ventes et profits par région
region_sales_profit = df_clean.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()


# In[62]:


# Identifier les régions avec des profits négatifs ou faibles
low_profit_regions = region_sales_profit[region_sales_profit['Profit'] <= 0]

# Afficher les résultats
print("Régions sous-performantes :")
print(low_profit_regions)


# In[63]:


# Graphique en barres pour les régions avec profits négatifs
fig = px.bar(
    low_profit_regions,
    x='Region',
    y='Profit',
    title="Régions Sous-Performantes (Profits Négatifs)",
    text='Profit',
    labels={'Profit': 'Profits', 'Region': 'Région'},
    color='Profit',
    color_continuous_scale='Reds'
)
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(template="plotly_white")
fig.show()


# ## Analyse du Graphique : Marges Bénéficiaires par Région
# 
# # Contexte
# **Le graphique montre les marges bénéficiaires par région, exprimées en pourcentage et visualisées avec des barres colorées représentant leur intensité.**
# 
# Analyse
# 
# - **Central :** Marge Bénéficiaire Négative **(-189,112.51%)** **texte en gras**
# 
#   La région Central est en difficulté, avec une marge négative.
# Cela signifie que les pertes dans cette région dépassent largement les ventes, ce qui indique un problème opérationnel ou une structure de coûts élevée.
# 
#   **Action recommandée :**
# Analyser les produits ou segments spécifiques responsables des pertes.
# Réduire les remises excessives ou optimiser les coûts.
# 
# 
# - **East : Marge Positive (190,462.61%)**
# 
#   La région East montre une bonne rentabilité.
# La marge est élevée, ce qui pourrait indiquer des coûts maîtrisés ou des remises modérées.
# 
#  **Action recommandée :**
# Maintenir la stratégie actuelle.
# Identifier les segments qui contribuent à ce succès pour répliquer dans d'autres régions.
# 
# 
# - **South : Marge Exceptionnellement Élevée (901,700.18%)**
# 
#   La région South est extrêmement rentable, avec une marge bénéfique hors norme.
# Une marge aussi élevée pourrait indiquer une concentration de ventes sur des produits à haute marge ou une sous-évaluation des coûts associés.
# 
# 
# - **West : Marge Élevée (752,022.55%)**
# 
#   La région West est également très performante, avec une marge significative.
# Cela montre une gestion efficace des ventes et des coûts.
# 
#  **Action recommandée :**
# Identifier les segments ou produits responsables de cette marge et renforcer la stratégie existante.
# 
# # Points Clés
# # *Problème Identifié :*
# 
# La région Central représente une source majeure de pertes, nécessitant une attention immédiate.
# 
# Régions Rentables :South et West sont les moteurs financiers de l'entreprise, avec des marges très élevées.
# 

# In[64]:


# Marges par catégorie
category_margins = df_clean.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
category_margins['Profit Margin (%)'] = (category_margins['Profit'] / category_margins['Sales']) * 100

# Graphique interactif
fig = px.bar(
    category_margins,
    x='Category',
    y='Profit Margin (%)',
    title="Marges Bénéficiaires par Catégorie",
    text='Profit Margin (%)',
    labels={'Profit Margin (%)': 'Marge (%)', 'Category': 'Catégorie'},
    color='Profit Margin (%)'
)
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig.update_layout(template="plotly_white")
fig.show()


# # **Analyse du Graphique : Marges Bénéficiaires par Catégorie**
# 
# # Observations :
# 
# - **Office Supplies :**
# 
#    Cette catégorie affiche la marge bénéficiaire la plus élevée **(773,739.89%).**
# Cela indique une rentabilité exceptionnelle pour les fournitures de bureau.
# 
# - **Technology :**
# 
#   La catégorie Technologie suit avec une marge bénéficiaire de **663,965.32%**.
# Cela reflète un bon équilibre entre coût et prix de vente pour les produits technologiques.
# 
# - **Furniture :**
# 
#   La catégorie des Meubles a une marge bénéficiaire bien plus faible **(17,418.40%).**
# Cela suggère des défis de rentabilité pour cette catégorie, potentiellement liés à des coûts élevés (fabrication, transport) ou à des remises excessives.
# 
# # Insights Clés :
#  **Office Supplies** et **Technology** sont les moteurs financiers, générant des marges élevées.
# 
#  **Furniture** est un point faible dans la rentabilité, nécessitant une stratégie d'amélioration.
# 
# Actions Recommandées :
# 
# - Prioriser les Catégories Rentables :
# Augmenter les investissements dans Office Supplies et Technology pour maintenir leur croissance.
# 
# - Optimiser les Meubles :
# Réduire les coûts (production, logistique).
# Réévaluer les remises pour éviter une baisse de rentabilité.

# # **Étape 5 : Synthèse des Résultats**
# 
# # Plan des Analyses
# 
# **Regrouper les insights clés de chaque analyse :**
# - Analyses Financières : Produits et catégories rentables, marges par région.
# - Analyses Saisonnières : Pics d’activité, périodes creuses, prévisions de ventes.
# - Analyses Géographiques : Régions performantes, zones sous-performantes, opportunités.
# 
# **Recommandations Stratégiques**
# 
# - Produits et Catégories :
#   Prioriser les produits à forte marge.
#   Optimiser les produits sous-performants.
# - Régions :
#   Renforcer les stratégies dans les régions rentables.
#   Implémenter des actions spécifiques dans les régions sous-performantes.
# - Planification Saisonnière :
#   Anticiper les pics de demande pour ajuster les stocks et les campagnes marketing.
#    Lancer des promotions pendant les périodes creuses.
# - Optimisation Globale :
#   Ajuster les remises pour maximiser les marges bénéficiaires.
# Réduire les coûts dans les zones sous-performantes.
# 
# **Livrables**
# 
# Document de synthèse :
# 
# Résumer les insights et les recommandations sous forme structurée.
# 
# Plan d’action :
# Proposition des initiatives concrètes pour les aspects financiers, saisonniers, et géographiques.

# # **Synthèse des Résultats**
# 
# **1. Analyses Financières**
# 
#     - Produits Performants :
# Les produits technologiques dominent en termes de ventes et profits.
# Les fournitures de bureau affichent une marge bénéficiaire (773,739.89%).
# 
#     - Catégories Sous-Performantes :
# La catégorie Furniture est la moins rentable (marge 17,418.40%) en raison de coûts élevés ou de remises excessives.
# 
#     - Régions Rentables :
# West et South affichent des profits élevés, avec des marges respectives de 752,022.55% et 901,700.18%.
# 
#     - Régions Sous-Performantes :
# Central est en difficulté, enregistrant des pertes significatives.
# 
# 2. Analyses Saisonnières
# 
#     - Les ventes suivent une tendance générale à la hausse, avec des pics récurrents en fin d'année (novembre et décembre).
#     - Les périodes creuses incluent les débuts d'année (janvier à mars).
#     - La moyenne glissante confirme une croissance stable malgré des fluctuations.
#     
# 3. Analyses Géographiques
# 
#     - Régions Rentables :
#      West et South sont les principales zones de profit.
#      
#     - Zones Sous-Performantes :
# Les pertes dans la région Central nécessitent des actions ciblées.
# 
#    - Les cartes choroplèthes mettent en évidence des zones géographiques prioritaires pour améliorer les performances
# 
# **Insights Clés**
# 
#     - Rentabilité : Office Supplies et Technology sont les catégories les plus rentables.
#     - Pertes Régionales : La région Central souffre de pertes significatives.
#     - Saisonnalité : Les pics de ventes en fin d'année peuvent être exploités davantage
# 
# # **Recommandations Stratégiques**
# 
# **1. Produits et Catégories**
# 
# Prioriser les catégories rentables :
# 
#     - Investir davantage dans Office Supplies et Technology, qui génèrent des marges élevées.
#     - Optimiser les Meubles :
# 
#   Réduire les coûts de production et de logistique.
#   Réévaluer les politiques de remises pour éviter une érosion des marges.
# 
# **2. Régions**
# 
#     - Renforcer les zones performantes :
# Consolider les stratégies dans les régions West et South pour maintenir leur profitabilité.
# 
#     - Redresser les zones sous-performantes :
# Mettre en œuvre des plans d'optimisation dans la région Central :
# 
#     - Réduire les coûts opérationnels.
# et lancer des campagnes promotionnelles ciblées pour stimuler les ventes.
# 
# **3. Planification Saisonnière**
# 
#     - Exploiter les pics de fin d’année :
# Renforcement des campagnes marketing pendant novembre et décembre.
# 
#     - Augmenter les stocks des produits performants pour répondre à la demande.
# 
#     -Stimuler les périodes creuses :
# Lancer des promotions ou des offres spécifiques entre janvier et mars.
# 
# **4. Optimisation Globale**
# 
#     - Réduction des remises excessives : Mettre en place des seuils de remises pour éviter une baisse de rentabilité.
#     - Suivi des performances : Implémenter des tableaux de bord pour surveiller les ventes, profits, et marges en temps réel.
# 
# # **Exportation du dataset**

# In[ ]:


from google.colab import files

# Chemin pour sauvegarder les données nettoyées
cleaned_file_path = '/content/cleaned_data.csv'

# Exporter le DataFrame nettoyé sous forme de fichier CSV
df.to_csv(cleaned_file_path, index=False)

# Télécharger le fichier localement
files.download(cleaned_file_path)


# ## **PARTIE MODELISATION**
# 
# # **Phase 6 : Création Modèle de ML**
# 
# # **Regression Lineaire**
# 
# **Objectifs :**
# 
#   - Prédire le montant des ventes pour une période donnée en fonction des caractéristiques disponibles dans les données.
# 
# **Approche technique :**
# 
# Variables explicatives (à determiner après analyse)
# 
#   - Variables quantitatives : coût, quantité commandée, escompte (discount).
#   - Variables catégoriques : région, catégorie, sous-catégorie.
# 
# Métriques d’évaluation  
# 
#   - RMSE (Root Mean Squared Error) : mesure l’erreur moyenne de prédiction.
#   - R² (coefficient de détermination) : évalue la proportion de variance expliquée par le modèle.
#   - MAE (Mean Absolute Error) : mesure l’écart absolu moyen.
# 
# 
# 
# Optimiser les hyperparamètres du meilleur modèle
# - (GridSearchCV).
# 
# 
# Pipeline  
# 
#   - Prétraitement : encodage des variables catégoriques, normalisation des variables numériques.
#   - Séparation des ensembles : 80% entraînement, 20% test.
# 
# 
# 
#   # **Classification pour segment : Regression Logistisque**
# 
# **Objectifs :**
# 
#   - Classifier les clients en différents segments (clients réguliers, clients premium, clients inactifs) en fonction de leurs comportements d’achat.
# 
# **Approche technique :**
# 
# Variables explicatives (a determiner après analyse)
# 
#   - Total des ventes par client.
#   - Fréquence des achats.
#   - Montant moyen des transactions.
# 
# Métriques d’évaluation  
# 
#   - Accuracy : proportion des prédictions correctes.
#   - F1-Score : combine précision et rappel
#   - Confusion Matrix : pour analyser les erreurs de classification.
# 
# Pipeline  
# 
#   - Prétraitement : normalisation, gestion des données manquantes.
#   - Entraînement et validation : K-fold cross-validation pour éviter l’overfitting.

# In[65]:


# Renommer les colonnes pour supprimer les espaces
df_clean.columns = [col.replace(' ', '_') for col in df_clean.columns]


# # **Sélection et préparation des données**
# 
# **Tâches :**
# 
#   - Identifier les colonnes pour la régression et la classification.
# 
# Sélection des colonnes pertinentes :
# 
# 1. Identifier les colonnes explicatives (features) et la colonne cible (target) pour chaque objectif.
# 
# 
# Effectuer le prétraitement nécessaire :
#       
# 1.   Encodage des colonnes catégoriques.
# 2.   Normalisation des colonnes numériques.
# 3.   Division des données en ensembles d'entraînement (80%) et de test (20%).
# 
# - **Livrable attendu :**
#  1. Jeu de données prêt pour la modélisation
# 
# # **CHOIX DES VARIABLES SIGNIFICATIVES**
# 
# - **Regression Lineaire**
#   Nous avons la matrice de corrélation que nous avons effectuer plus haut. Nous avons constater que seuls les variables Quantity et Discount ont une corrélation moindre significative avec notre variable cible sales. Donc nous allons les utiliser dans notre modèle.
# 
# - **Classification (Régression logistique) **
# 
# Pour cela nous allons effectuer les test de ANOVA ET CHI2 pour determiner la P-value des variable avec notre variable cible segment.
# 

# In[66]:


# Créer une copie pour chaque modèle
df_logistic = df_clean.copy()  # Pour la régression logistique
df_linear = df_clean.copy()    # Pour la régression linéaire
df_rf = df_clean.copy()      # Pour Random Forest
df_knn = df_clean.copy()     # Pour K-Nearest Neighbors
df_dt = df_clean.copy()   # Pour Arbre de décision
df_svm = df_clean.copy()  # Pour SVM


# In[67]:


#importation des bibliothèques
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import GridSearchCV
import plotly.graph_objects as go
import sklearn


# # **Regression Lineaire**
# 
# **Tâches :**
# 
#   - Charger les données et examiner leur structure.
#   - Identifier les problèmes de qualité des données.
#   - Réaliser une description statistique initiale (moyennes, médianes, écart-types).
# 
# **Livrables :**
# 
#   - Rapport d’audit des données (problèmes identifiés, suggestions de nettoyage).

# In[69]:


# Variables explicatives et cible
features = ['Quantity', 'Discount']
target = 'Sales'

X = df_linear[features]
y = df_linear[target]


# In[70]:


# Vérifier les données manquantes et les traiter
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    X = X.fillna(0)
    y = y.fillna(0)


# In[71]:


# 2. Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)


# In[73]:


# Évaluation du modèle et Prédictions
y_pred = model.predict(X_test)


# In[74]:


# Calcul des métriques
r2 = r2_score(y_test, y_pred)
rmse =np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)


# In[75]:


# Affichage des métriques
print(f"R² : {r2:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE : {mae:.2f}")


# In[76]:


# Visualisation des résultats
# Scatter plot des valeurs réelles vs prédictions
scatter_fig = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Valeurs Réelles', 'y': 'Prédictions'},
    title="Valeurs Réelles vs Prédictions (Régression Linéaire)"
)
scatter_fig.update_traces(marker=dict(size=5, opacity=0.7))
scatter_fig.show()


# In[77]:


# Visualisation des résidus
residuals = y_test - y_pred
residual_fig = px.scatter(
    x=y_pred,
    y=residuals,
    labels={'x': 'Prédictions', 'y': 'Résidus'},
    title="Analyse des Résidus (Régression Linéaire)"
)
residual_fig.update_traces(marker=dict(size=5, opacity=0.7))
residual_fig.show()


# # **Interprétation des Résultats du Modèle de Régression Linéaire**
# 
#     **R² (0.02) :** signifie que seulement 2% de la variance des ventes (Sales) est expliquée par les variables explicatives (Quantity et Discount).
#     Cela indique que le modèle est incapable de capturer correctement la relation entre les variables explicatives et la cible.
# 
#     **RMSE (414,497.70) :** L'erreur quadratique moyenne est très élevée, indiquant une grande différence entre les prédictions du modèle et les valeurs réelles. Une valeur aussi importante reflète un modèle avec de faibles performances prédictives.
# 
#     **MAE (144,860.17) :** L'erreur absolue moyenne, bien qu'inférieure au RMSE, reste élevée, ce qui confirme que les prédictions sont souvent éloignées des valeurs réelles.

# # **Regression Logistique**
# 
# **Tâches :**
# 
#   - Régression logistique.
#   - KNN (k-nearest neighbors).
#   - Arbre de décision.
#   - Random Forest.
#   - SVM (Support Vector Machine).
# 
# **Livrables :**
# 
#   - Rapport d’audit des Comparaison des modèles
#   - Comparer les performances des modèles à l’aide des métriques mentionnées.
#   - Présenter les résultats sous forme de tableaux ou de graphiques (éventuellement des bar plots).

# In[78]:


#Analyse catégorielle : Analysez l'influence des variables catégoriques
fig = px.box(df_clean, x='Region', y='Sales', color='Category')
fig.show()


# In[79]:


# Création du boxplot interactif avec Plotly
fig = px.box(
    df_clean,
    x="Segment",
    y="Profit",
    title="Relation entre Profit et Segment",
    labels={"Profit": "Profit ($)", "Segment": "Segment Client"},
    color="Segment",  # Ajoute des couleurs différentes pour chaque segment
    template="plotly_white"
)


# In[80]:


# Afficher le graphique
fig.update_layout(
    xaxis_title="Segment",
    yaxis_title="Profit ($)",
    title_font_size=20
)
fig.show()


# In[81]:


# Initialisation des listes pour stocker les résultats
chi2_results = []
anova_results = []


# In[82]:


# Séparer les variables catégoriques et continues
categorical_vars = df_clean.select_dtypes(include='object').columns
continuous_vars = df_clean.select_dtypes(include=['int64', 'float64']).columns


# In[83]:


# Tester les variables catégoriques (Chi-2)
for var in categorical_vars:
    if var != 'Segment':  # Exclure la variable cible
        contingency_table = pd.crosstab(df_clean['Segment'], df_clean[var])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi2_results.append({
            "Variable": var,
            "Test": "Chi-2",
            "Statistique": chi2,
            "p-value": p,
            "Significative": "Oui" if p <= 0.05 else "Non"
        })


# In[84]:


# Tester les variables continues (ANOVA)
for var in continuous_vars:
    if var != 'Segment':  # Exclure la variable cible
        anova_result = f_oneway(
            *(df_clean[df_clean['Segment'] == segment][var] for segment in df_clean['Segment'].unique())
        )
        anova_results.append({
            "Variable": var,
            "Test": "ANOVA",
            "Statistique": anova_result.statistic,
            "p-value": anova_result.pvalue,
            "Significative": "Oui" if anova_result.pvalue <= 0.05 else "Non"
        })


# In[85]:


# Combinaison des résultats
results_df = pd.DataFrame(chi2_results + anova_results)


# In[86]:


# Afficher les résultats dans un tableau interactif avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=["Variable", "Test", "Statistique", "p-value", "Significative"]),
    cells=dict(values=[results_df.Variable, results_df.Test, results_df.Statistique, results_df["p-value"], results_df.Significative],
               align=['left'] * len(results_df.columns))
)])
fig.show()


# # **INTERPRETATIONS DES RESULTATS DES TESTS ANOVA ET CHI2**
# 
# **Interprétation des résultats Chi-2 :**
# 
# - Les variables catégoriques significatives pour **Segment (p-value ≤ 0.05)** sont :
# Order_ID, Ship_Mode, Customer_ID, Customer_Name, City, State, Postal_Code
# Product_ID, Product_Name, Order_Week.
# 
# **Interprétation des résultats ANOVA :**
# 
# - Les variables continues significatives pour **Segment (p-value ≤ 0.05)** sont :  Row_ID

# #Pretraitement des données
# #Utilisation du dataset df_logistic pour regression logistic
# #import plotly.express as px

# In[87]:


# Sélectionner les colonnes pour la visualisation
visualization_features = ['Ship_Mode', 'City', 'State']
df_viz = df_logistic[visualization_features + ['Segment']]


# In[88]:


# Encodage des variables catégoriques pour la visualisation
encoded_df = df_viz.copy()
for col in visualization_features:
    if encoded_df[col].dtype == 'object':
        encoded_df[col] = encoded_df[col].astype('category').cat.codes


# In[89]:


# 1. Scatter plots en paires (scatter matrix)
fig_scatter = px.scatter_matrix(
    encoded_df,
    dimensions=visualization_features,
    color='Segment',  # Colorer selon la variable cible
    title="Scatter Matrix des Variables de Régression Logistique",
    labels={col: col for col in visualization_features + ['Segment']},
    template="plotly_white"
)
fig_scatter.update_traces(diagonal_visible=False)  # Masquer les histogrammes diagonaux


# In[91]:


# 2. Histogramme des fréquences pour chaque variable
histograms = []
for feature in visualization_features:
    histograms.append(
        px.histogram(
            df_logistic, x=feature, color="Segment",
            title=f"Histogramme des Fréquences - {feature}",
            barmode='group',
            template="plotly_white"
        )
    )


# In[92]:


# 3. Graphes en courbes pour chaque variable (par exemple, count par segment et feature)
line_plots = []
for feature in visualization_features:
    grouped = df_logistic.groupby([feature, 'Segment']).size().reset_index(name='Count')
    line_plots.append(
        px.line(
            grouped, x=feature, y='Count', color='Segment',
            title=f"Graphe en Courbes - {feature} par Segment",
            markers=True,
            template="plotly_white"
        )
    )


# In[93]:


# Afficher les graphiques interactifs
print("Affichage des Scatter Plots en Paires :")
fig_scatter.show()

print("\nAffichage des Histogrammes :")
for hist in histograms:
    hist.show()

print("\nAffichage des Graphes en Courbes :")
for line in line_plots:
    line.show()


# # **CREATION ET ENTRAINNEMENT DU MODEL**

# In[94]:


# Variables indépendantes pour la classification
classification_features = ['Ship_Mode', 'City', 'State']
target_variable = 'Segment'


# In[95]:


# 1. Préparation des données
X = df_logistic[classification_features]
y = df_logistic[target_variable]


# In[96]:


# Encodage des variables catégoriques
#from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in classification_features:
    if X[col].dtype == 'object':  # Vérifier si la variable est catégorique
        X[col] = encoder.fit_transform(X[col])


# In[97]:


# 2. Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[98]:


# 3. Entraînement du modèle de régression logistique
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)


# In[99]:


# 4. Évaluation du modèle
# Prédictions
y_pred = model.predict(X_test)


# In[100]:


# Affichage dans un tableau
#print("Rapport de Classification :")
#print(report_df)


# Rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

# Visualisation avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(report_df.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[report_df.reset_index()[col] for col in report_df.reset_index().columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Rapport de Classification Regression Logistiques")
fig.show()


# In[101]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de Confusion :")
conf_matrix_df = pd.DataFrame(conf_matrix)  # Créez un DataFrame pour la matrice
print(conf_matrix)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()


# # **Analyse des résultats**
# 
# **Matrice de confusion** :
# La matrice de confusion montre les performances du modèle en termes de prédictions correctes et incorrectes pour chaque classe.
# 
# Consumer (Classe 1) :
# 
# 1019 instances ont été correctement classées comme "Consumer".
# Aucune instance de "Consumer" n'a été incorrectement classée dans les autres classes ("Corporate" ou "Home Office").
# 
# Corporate (Classe 2) :
# 
# 598 instances de "Corporate" ont été mal classées comme "Consumer".
# Aucune instance de "Corporate" n'a été correctement classée.
# 
# Home Office (Classe 3) :
# 
# 381 instances de "Home Office" ont été mal classées comme "Consumer".
# Aucune instance de "Home Office" n'a été correctement classée.
# Rapport de classification
# 
# Le tableau montre les métriques clés pour évaluer le modèle : précision, recall, f1-score et support.
# 
# Consumer :
# 
# - **Précision : 0.51**. Cela signifie que 51 % des prédictions pour la classe "Consumer" étaient correctes.
# 
# - **Recall : 1.00.** Cela signifie que toutes les instances de "Consumer" ont été correctement prédites.
# 
# - **F1-Score : 0.6755.** Ce score combine précision et recall, reflétant une performance modérée.
# 
# Corporate et Home Office :
# 
# La précision, le recall et le F1-score sont tous 0. Cela signifie que le modèle n'a pas pu prédire correctement ces classes.
# 
# - Accuracy (Précision globale) : 0.51. Le modèle a correctement classé 51 % des instances.
# 
# - Macro Average : Moyenne des métriques pour toutes les classes (non pondérée par le support).
# 
# - F1-Score : 0.225. Cela reflète une faible performance globale.
# Weighted Average : Moyenne pondérée des métriques selon le support de chaque classe.
# 
# - F1-Score : 0.3445. Cela indique que la performance globale du modèle est limitée
# 
# # **Interprétation**
# Performance dominante pour "Consumer" :
# 
# Le modèle excelle uniquement pour la classe "Consumer", ce qui peut indiquer un déséquilibre des données ou une forte prédominance des exemples "Consumer".
# Mauvaise prédiction pour "Corporate" et "Home Office" :
# 
# Les classes "Corporate" et "Home Office" ne sont jamais correctement prédites, ce qui montre un problème de représentativité ou de séparation des données pour ces classes.
# Précision globale limitée :
# 
# Avec une précision globale de 51 %, le modèle est biaisé et ne parvient pas à généraliser pour toutes les classes.
# 
# Nous allons effectué un test d'éfficacité **GridSearch** et ensuite nous devons effectuer des test avec d'autres modèles ue sont le Random Forest et KNN.

# In[102]:


#from sklearn.model_selection import GridSearchCV
# Définir le modèle de base
logistic_model = LogisticRegression(max_iter=1000, random_state=42)


# In[103]:


# Définir la grille des hyperparamètres
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],  # Types de pénalisation
    'C': [0.01, 0.1, 1, 10, 100],  # Régularisation
    'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg'],  # Solveurs compatibles
    'class_weight': [None, 'balanced']  # Gestion des classes déséquilibrées
}


# In[104]:


# Initialiser le GridSearch
grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=5,  # Validation croisée sur 5 plis
    scoring='f1_weighted',  # Optimisation basée sur le score F1 pondéré
    verbose=1,  # Affichage des détails
    n_jobs=-1  # Utiliser tous les cœurs disponibles
)


# In[105]:


# Exécuter le GridSearch sur les données d'entraînement
grid_search.fit(X_train, y_train)


# In[106]:


# Meilleurs hyperparamètres
best_params = grid_search.best_params_
print("Meilleurs hyperparamètres :", best_params)


# In[107]:


# Meilleur modèle
best_model = grid_search.best_estimator_


# In[108]:


# Évaluation sur le jeu de test
y_pred = best_model.predict(X_test)


# In[109]:


# Rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)  # Arrondi à 2 décimales


# In[110]:


# Affichage interactif du rapport de classification avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(report_df.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[report_df.reset_index()[col] for col in report_df.reset_index().columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Rapport de Classification après Optimisation")
fig.show()


# In[111]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion après optimisation :")
print(conf_matrix)


# In[112]:


# Affichage interactif de la matrice de confusion avec Plotly
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=best_model.classes_,
    columns=best_model.classes_
)

fig_conf_matrix = go.Figure(data=[go.Table(
    header=dict(values=[''] + list(conf_matrix_df.columns),
                fill_color='paleturquoise',
                align='center'),
    cells=dict(values=[conf_matrix_df.index] + [conf_matrix_df[col] for col in conf_matrix_df.columns],
               fill_color='lavender',
               align='center'))
])

fig_conf_matrix.update_layout(title="Matrice de Confusion Interactif")
fig_conf_matrix.show()


# # **MODEL RANDOM FOREST**
# 
# Résumé du pipeline complet :
# - Encodage des variables catégoriques (Ship_Mode, City, State).
# - Ajout des variables numériques (Quantity, Discount, Profit).
# - Séparation des données en ensembles d'entraînement et de test.
# - Entraînement du modèle avec RandomForestClassifier.
# - Évaluation des performances via le rapport de classification et la matrice de confusion.
# - Analyse de l’importance des variables pour identifier celles qui contribuent le plus au modèle.
# 
# Bien que certaine svariables numériques n'ont pas de correlation avec la variable cible, nous allons les intégrer pour l'évaluation du modèle.
# 
# Variables supplémentaires sélectionnées :
# 
# - Quantity : Reflète la quantité commandée ; elle peut indirectement indiquer le segment client.
# - Discount : Peut influencer le comportement d'achat et le segment.
# - Profit : Bien que sa corrélation soit faible, il peut capturer des tendances spécifiques.
# 

# In[114]:


#Le modèle Random Forest nécessite que toutes les variables soient numériques.
#Nous devons donc encoder les variables catégoriques avec Label Encoding.
categorical_columns = ['Ship_Mode', 'City', 'State']
for col in categorical_columns:
    df_rf[col] = df_rf[col].astype('category').cat.codes


# In[115]:


#Séparation des données et entrapinnement du modèle
#from sklearn.ensemble import RandomForestClassifier

# Variables explicatives et cible
X = df_rf[['Ship_Mode', 'City', 'State', 'Quantity', 'Discount', 'Profit']]
y = df_rf['Segment']


# In[116]:


# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[117]:


# Entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[118]:


#Evaluation du modele
# Prédictions
y_pred = model.predict(X_test)


# In[119]:


# Rapport de classification
print("Rapport de classification :")

# Obtenir le rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)

# Convertir le rapport en DataFrame Pandas
df_report = pd.DataFrame(report).transpose().round(2)

# Afficher le DataFrame
#display(df_report)

# Visualisation avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_report.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_report.reset_index()[col] for col in df_report.reset_index().columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Rapport de Classification Random Forest")
fig.show()


# In[120]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(conf_matrix)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()


# In[121]:


# Importance des features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Importance des variables :")
print(feature_importance)


# Visualisation de l'importance
feature_importance.plot(kind='bar', x='Feature', y='Importance', title="Importance des Variables")
plt.show()


# # **Interprétation des résultats :**
# 
# Rapport de classification :
# 
# **Consumer (Classe majoritaire) :**
# Bonne précision (0.54) et rappel (0.69), indiquant que le modèle détecte bien cette classe.
# Le f1-score (0.60) montre une performance globale correcte pour cette classe.
# 
# **Corporate et Home Office (Classes minoritaires) :**
# Les scores sont faibles (précision, rappel, f1-score), montrant des difficultés du modèle à bien les prédire.
# Accuracy globale (47%) :
# 
# Le modèle est biaisé vers la classe majoritaire (Consumer), ce qui réduit son efficacité pour les autres classes.
# 
# **Matrice de confusion :**
# 
# Le modèle prédit bien la classe Consumer (705 bonnes prédictions).
# Les autres classes (Corporate et Home Office) sont souvent mal classées, ce qui souligne un déséquilibre dans les performances.
# 
# 
# **Importance des variables :**
# 
# - Profit (0.45) : Variable la plus importante, indiquant qu'elle influence fortement les décisions du modèle.
# 
# - City (0.20) : Influence modérée, montrant que la localisation joue un rôle important dans la classification.
# 
# - Quantity (0.13) : Reflète un impact significatif, mais moindre que les deux premières.
# 
# - State (0.10) : Influence relativement faible, mais utile pour la classification.
# 
# - Ship_Mode (0.06) et Discount (0.04) : Variables moins influentes, contribuent peu au modèle.
# 
# 
# # **Conclusion :**
# 
# Le modèle est biaisé en faveur de la classe majoritaire(Consumer).
# Les variables les plus importantes (Profit, City, et Quantity) peuvent être exploitées davantage dans l'analyse ou pour optimiser le modèle.

# # **MODEL KNN**
# 
# Nous allons utiliser les mêmes variable que celles du modèle Random Forest pour garder la cohérence de notre projet afin de pouvoir faire une évaluation concrète des performances de chaque modèle et selectionner le meilleur modèle.
# 
# Comme pour le Random Forest toutes le svariables doivent être numérique.
# - Normalisation de toutes les variables explicatives.
# - Encodage des variables catégoriques en entiers.

# In[122]:


#from sklearn.preprocessing import MinMaxScaler
# 1. Préparation des données
df_knn = df_clean.copy()


# In[123]:


# Sélection des variables
knn_features = ['Profit', 'City', 'Quantity', 'State', 'Ship_Mode', 'Discount']
target_variable = 'Segment'


# In[124]:


# Encodage des variables catégoriques
encoder = LabelEncoder()
for col in ['City', 'State', 'Ship_Mode']:
    df_knn[col] = encoder.fit_transform(df_knn[col])


# In[125]:


# Normalisation des variables explicatives
scaler = StandardScaler()
df_knn[knn_features] = scaler.fit_transform(df_knn[knn_features])


# In[126]:


# Séparer les variables explicatives (X) et la cible (y)
X = df_knn[knn_features]
y = df_knn[target_variable]


# In[127]:


# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[128]:


#from sklearn.neighbors import KNeighborsClassifier
# 2. Entraînement du modèle KNN
knn_model = KNeighborsClassifier(n_neighbors=10)  # k = 5 par défaut
knn_model.fit(X_train, y_train)


# In[129]:


# 3. Évaluation du modèle
# Prédictions
y_pred = knn_model.predict(X_test)


# In[130]:


# Rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)

# Convertir le rapport en DataFrame Pandas et arrondir les valeurs
df_report = pd.DataFrame(report).transpose().round(2)

# Visualisation interactive du rapport de classification avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_report.reset_index().columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')),  # Personnalisation des entêtes
    cells=dict(values=[df_report.reset_index()[col] for col in df_report.reset_index().columns],
               fill_color='lavender',
               align='left',
               font=dict(size=11, color='black'))  # Personnalisation des cellules
)])

fig.update_layout(title="Rapport de Classification KNN")
fig.show()

#print("Rapport de classification :")
#print(classification_report(y_test, y_pred))


# In[131]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(conf_matrix)

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()


# # **Analyse et interprétation des résultats pour le modèle KNN :**
# 
# Rapport de classification :
# Comme pour les modèle précédents nous constatons que KNN classifie correctement le segment Consumer et a du ùmal a classifier les autres segments.
# 
# Avec :
# - Précision globale (Accuracy) : 0.48,le modèle classifie correctement environ 48 % des échantillons.
# 
# 
# **Matrice de confusion :**
# 
# Interprétation générale :
# 
# - Points forts :
# Bonne performance pour la classe "Consumer" (précision et rappel acceptables).
# 
# - Points faibles :Les classes "Corporate" et "Home Office" sont mal distinguées, avec de faibles rappels et F1-scores.
# Déséquilibre des performances entre les classes.
# 
# Conclusion : On constate que le modèle KNN montre une tendance à bien classer la classe majoritaire ("Consumer") mais a des difficultés pour les classes minoritaires. Cela pourrait être amélioré en ajustant les hyperparamètres (nombre de voisins k)

# # **MODEL ARBRE DE DECISION**
# 
# Nous utiliserons les variables significatives identifiées précédemment pour la classification.
# Les données catégoriques seront encodées en numérique, et nous utiliserons uniquement des données numériques pour l'arbre de décision.

# In[132]:


#from sklearn.tree import DecisionTreeClassifier, plot_tree
# Étape 1 : Préparation des données
categorical_columns = ['Ship_Mode', 'City', 'State']


# In[133]:


# Encodage des variables catégoriques
for col in categorical_columns:
    df_dt[col] = df_dt[col].astype('category').cat.codes


# In[134]:


# Définir les variables explicatives et la cible
X = df_dt[['Ship_Mode', 'City', 'State', 'Profit', 'Quantity', 'Discount']]
y = df_dt['Segment']


# In[135]:


# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[136]:


# Étape 2 : Entraînement du modèle
tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_model.fit(X_train, y_train)


# In[137]:


# Étape 3 : Évaluation du modèle
# Prédictions
y_pred = tree_model.predict(X_test)


# In[138]:


# Rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose().round(2)

# Visualisation du rapport avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_report.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_report.reset_index()[col] for col in df_report.reset_index().columns],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title="Rapport de Classification - Arbre de Décision")
fig.show()


# In[139]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tree_model.classes_, yticklabels=tree_model.classes_)
plt.title("Matrice de Confusion - Arbre de Décision")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()


# In[140]:


# Étape 4 : Importance des variables
importances = tree_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Visualisation des importances
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Importance des Variables - Arbre de Décision")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.show()


# In[141]:


# Étape 5 : Affichage de l'arbre de décision
plt.figure(figsize=(20, 10))  # Ajuster la taille pour un meilleur affichage
plot_tree(tree_model, feature_names=X.columns, class_names=tree_model.classes_, filled=True, rounded=True)
plt.title("Visualisation de l'Arbre de Décision")
plt.show()


# # **Analyse des Résultats du Modèle Arbre de Décision**
# 
# Rapport de Classification :
# 
# Consumer : La précision est de 0.51 avec un rappel élevé de 0.99, ce qui indique que le modèle classe presque tous les échantillons de "Consumer" correctement, mais avec quelques erreurs de précision.
# 
# Corporate : La précision est de 0.56, mais le rappel est très faible (0.03), signifiant que très peu d'échantillons "Corporate" sont correctement identifiés.
# 
# Home Office : Bien que la précision atteigne 1, le rappel est extrêmement faible (0.01), ce qui montre que le modèle ne capture presque pas cette catégorie.
# 
# 
# **Résumé :** L'accuracy globale est de 52%, mais le score macro moyen (0.34) montre un déséquilibre dans les performances entre les classes.
# 
# **Matrice de Confusion :**
# 
# Le modèle montre une forte tendance à prédire la classe "Consumer", avec 816 prédictions correctes sur 1019.
# Comme celles précédentes elles classes mal les autres segments
# 
# 
# Importance des Variables :
# 
# La variable "City" est de loin la plus influente dans les décisions de classification avec une importance de 0.5.
# 
# Les variables "Quantity" et "State" viennent ensuite, avec des importances modérées (~0.2 et ~0.1 respectivement).
# 
# Les autres variables (Profit, Discount, Ship_Mode) ont une importance beaucoup plus faible, montrant qu'elles contribuent peu à la classification.
# 
# 
# **Arbre de Décision :**
# 
# La visualisation de l'arbre montre une hiérarchie claire, où "City" est souvent utilisée comme critère de décision initial.
# Les feuilles montrent que le modèle segmente les données en utilisant des seuils simples, mais certaines branches sont sous-exploitées, conduisant à des performances limitées pour les classes minoritaires.
# 
# 
# **L'arbre de décision est biaisé en faveur de la classe dominante ("Consumer"), tandis que les autres classes sont mal représentées**.

# **MODEL SVM**
# 
# Étapes :
# - Encodage des variables catégoriques pour transformer les données en format numérique.
# - Normalisation des variables explicatives pour que SVM fonctionne efficacement (car il est sensible aux différences d'échelles).

# In[142]:


#from sklearn.svm import SVC
# Variables significatives
classification_features = ['Ship_Mode', 'City', 'State', 'Profit', 'Quantity']
target_variable = 'Segment'


# In[143]:


# Encodage des variables catégoriques
encoder = LabelEncoder()
for col in classification_features:
    if df_svm[col].dtype == 'object':  # Vérifier si la variable est catégorique
        df_svm[col] = encoder.fit_transform(df_svm[col])


# In[144]:


# Normalisation des variables numériques
scaler = StandardScaler()
df_svm[classification_features] = scaler.fit_transform(df_svm[classification_features])


# In[145]:


# Séparation des données
X = df_svm[classification_features]
y = df_svm[target_variable]


# In[146]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[147]:


# Étape 2 : Entraînement du modèle SVM
svm_model = SVC(kernel='linear', random_state=42)  # Kernel linéaire par défaut
svm_model.fit(X_train, y_train)


# In[148]:


# Étape 3 : Évaluation du modèle
y_pred = svm_model.predict(X_test)


# In[152]:


# Rapport de classification
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Affichage interactif avec Plotly
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_report.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df_report.reset_index()[col] for col in df_report.reset_index().columns],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(title="Rapport de Classification - SVM")
fig.show()


# In[151]:


# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Affichage interactif avec Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.title("Matrice de Confusion - SVM")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()


# In[150]:


# Vérifier que le kernel est linéaire
if svm_model.kernel == 'linear':
    # Extraire les coefficients (w)
    feature_importance = pd.DataFrame({
        'Feature': classification_features,
        'Coefficient': np.abs(svm_model.coef_[0])  # Utiliser les valeurs absolues des coefficients
    }).sort_values(by='Coefficient', ascending=False)

    # Visualisation des coefficients avec Matplotlib
    plt.figure(figsize=(6, 4))
    plt.bar(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
    plt.title("Importance des Variables - SVM (Kernel Linéaire)")
    plt.xlabel("Variables")
    plt.ylabel("Importance (Coefficient)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("L'importance des variables n'est pas disponible pour un kernel non-linéaire.")


# # **Performances globales :**
# 
# Le modèle SVM, avec un kernel linéaire, montre des performances faibles avec une précision élevée uniquement pour la classe "Consumer".
# 
# Les classes "Corporate" et "Home Office" ne sont pas correctement prédites (valeurs de rappel et de précision nulles).
# 
# **Importance des variables :**
# 
# Profit a la plus grande importance dans la classification, suivi de Ship_Mode.
# Les autres variables, telles que State, Quantity, et City, ont un impact limité sur les décisions.
# Conclusion :
# 
# Le modèle SVM est sous-optimal dans ce contexte, car il se concentre presque exclusivement sur la classe majoritaire "Consumer".

# ## **Comparaison des modèles**

# In[153]:


# Données des modèles
data = {
    "Model": [
        "Regression Logistique",
        "Rééquilibrage (SMOTE)",
        "Random Forest",
        "KNN",
        "Arbre de Décision",
        "SVM"
    ],
    "Accuracy": [0.51, 0.32, 0.47, 0.48, 0.52, 0.51],
    "Macro Avg F1-Score": [0.23, 0.27, 0.38, 0.33, 0.25, 0.23],
    "Weighted Avg F1-Score": [0.34, 0.31, 0.45, 0.42, 0.36, 0.34],
}

# Création du DataFrame
df_comparison = pd.DataFrame(data)

# Bar plot des métriques
fig = px.bar(
    df_comparison.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Comparaison des Performances des Modèles",
    labels={"Score": "Valeur", "Model": "Modèle", "Metric": "Métrique"},
    template="plotly_white"
)

fig.show()


# In[154]:


# Identifier le meilleur modèle
best_model = df_comparison.sort_values(by="Weighted Avg F1-Score", ascending=False).iloc[0]

best_model

# Compilation des métriques pour chaque modèle
metrics = {
    'Model': ['Logistic Regression', 'Random Forest', 'KNN', 'Decision Tree', 'SVM'],
    'Accuracy': [0.32, 0.47, 0.48, 0.52, 0.51],
    'Precision (macro avg)': [0.36, 0.41, 0.38, 0.69, 0.17],
    'Recall (macro avg)': [0.33, 0.38, 0.35, 0.34, 0.33],
    'F1-Score (macro avg)': [0.27, 0.38, 0.33, 0.25, 0.23]
}

# Création d'un DataFrame
df_metrics = pd.DataFrame(metrics)

# Création d'un bar plot pour comparer les métriques des modèles
fig = px.bar(
    df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value'),
    x='Model',
    y='Value',
    color='Metric',
    barmode='group',
    title="Comparaison des Performances des Modèles",
    labels={'Value': 'Score', 'Model': 'Modèles', 'Metric': 'Métriques'},
    template='plotly_white'
)

# Afficher le graphique
fig.show()


# In[155]:


from IPython.display import display
display(df_metrics) #
# Afficher le tableau des métriques
#import ace_tools as tools; tools.display_dataframe_to_user(name="Comparaison des Performances des Modèles", dataframe=df_metrics)


# In[156]:


import matplotlib.pyplot as plt

# Données pour les modèles
data = {
    "Model": [
        "Regression Logistique",
        "Rééquilibrage (SMOTE)",
        "Random Forest",
        "KNN",
        "Arbre de Décision",
        "SVM"
    ],
    "Accuracy": [0.51, 0.32, 0.47, 0.48, 0.52, 0.51],
    "Macro Avg F1-Score": [0.23, 0.27, 0.38, 0.33, 0.25, 0.23],
    "Weighted Avg F1-Score": [0.34, 0.31, 0.45, 0.42, 0.36, 0.34],
}

# Création d'un DataFrame
df_comparison = pd.DataFrame(data)

# Bar plots pour chaque métrique
metrics = ["Accuracy", "Macro Avg F1-Score", "Weighted Avg F1-Score"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(df_comparison["Model"], df_comparison[metric], color="skyblue")
    plt.title(f"Comparaison des Modèles - {metric}", fontsize=14)
    plt.xlabel("Modèles", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# In[157]:


# Identifier le meilleur modèle basé sur Weighted Avg F1-Score
best_model = df_comparison.loc[df_comparison["Weighted Avg F1-Score"].idxmax()]
best_model


# Meilleure précision pondérée : Random Forest avec 0.45.
# 
# Meilleure précision moyenne (Macro Avg) : Random Forest avec 0.38.
# 
# Meilleure précision globale (Accuracy) : Arbre de Décision avec 0.52.
# 
# Meilleur modèle :
# 
# **Modèle choisi : Random Forest**
# 
# - Justification : Offre un bon équilibre entre les différentes métriques (Weighted Avg F1-Score et Macro Avg F1-Score).
# Limites et axes d’amélioration :
# 
# - Déséquilibre des classes : Affecte fortement les modèles comme SVM et Logistique.
# 
# - Données catégoriques : Encodage simple peut perdre des informations cruciales.
# 
# **Axes d’amélioration :**
# Augmenter les données pour réduire le déséquilibre.
# Essayer des techniques avancées d'encodage (Target Encoding, Embeddings).
# Utiliser des modèles plus complexes comme Gradient Boosting.
# 
# # **Phase 6 : Développement d’une Application Web**
# L’objectif de cette phase est de transformer les analyses et visualisations en une application web interactive. Cette application permettra de :
# 
# - Visualiser les insights clés à l’aide de tableaux et graphiques interactifs.
# - Explorer dynamiquement les données pour une prise de décision plus rapide.

# In[162]:


# Installer les bibliothèques nécessaires
get_ipython().system('pip install streamlit')
get_ipython().system('pip install pyngrok')


# In[166]:


from pyngrok import ngrok

# Remplacez `YOUR_AUTH_TOKEN` par votre token
ngrok.set_auth_token("2rTWvQjPWhdRwoX79HSzMs2wLpK_kUXemf3fS6JiLuRPbL3Z")


# In[ ]:


get_ipython().system('ngrok http status')


# In[ ]:


get_ipython().system('pip install streamlit')


# In[ ]:




