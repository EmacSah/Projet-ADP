import streamlit as st
import pandas as pd
import numpy as np
#import plotly.express as px
from pathlib import Path
import os
import sys


# Ajout du chemin du projet au PYTHONPATH
#file_path = Path(__file__).parent.resolve()
#sys.path.append(str(file_path))

current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

#from src.data_processing import load_data, clean_data, prepare_model_data
#from src.visualization import (
 #	 create_sales_by_region, create_profit_by_category,
  #	 create_monthly_sales_trend, create_segment_distribution
#)
#from src.modeling import SuperstoreModel
#from src.utils import create_kpi_metrics, create_confusion_matrix_plot, download_predictions


# Gestion de l'importation de plotly
try:
	import plotly.express as px
	import plotly.graph_objects as go
except ImportError:
	st.error("Erreur lors de l'importation de plotly. Installation des dépendances...")
	os.system('pip install plotly==5.18.0 plotly-express==0.4.1 kaleido==0.2.1')
	import plotly.express as px
	import plotly.graph_objects as go

try:
	from src.dependencies import import_dependencies
	deps = import_dependencies()
	
	# Import des modules après vérification
	pd = deps['pd']
	np = deps['np']
	px = deps['px']
	RandomForestClassifier = deps['RandomForestClassifier']
	from src.data_processing import load_data, clean_data, prepare_model_data
	from src.visualization import (
		create_sales_by_region, create_profit_by_category,
		create_monthly_sales_trend, create_segment_distribution
	)
	from src.modeling import SuperstoreModel
	from src.utils import create_kpi_metrics, create_confusion_matrix_plot, download_predictions
except Exception as e:
	st.error(f"Erreur d'importation : {e}")
	st.stop()



# Configuration de la page
st.set_page_config(
	page_title="Superstore Analytics Dashboard",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_cached_data():
	try:
		df = load_data()
		df_clean = clean_data(df)
		return df_clean
	except Exception as e:
		st.error(f"Erreur lors du chargement des données: {e}")
		return None

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
	"Choisir une page",
	["Accueil", "Analyse Exploratoire", "Modélisation", "Prédictions"]
)

# Chargement des données
df = load_cached_data()

if df is not None:
	if page == "Accueil":
		st.title("📊 Dashboard Superstore Analytics")
		st.write("Bienvenue dans l'application d'analyse des données Superstore")
		
		# KPIs
		kpis = create_kpi_metrics(df)
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Total des Ventes", kpis['Total Sales'])
		with col2:
			st.metric("Total des Profits", kpis['Total Profit'])
		with col3:
			st.metric("Marge de Profit", kpis['Profit Margin'])
	 

	elif page == "Analyse Exploratoire":
		st.title("🔍 Analyse Exploratoire des Données")
		
		# Visualisations
		col1, col2 = st.columns(2)
		with col1:
			st.plotly_chart(create_sales_by_region(df), use_container_width=True)
			st.plotly_chart(create_monthly_sales_trend(df), use_container_width=True)
		with col2:
			st.plotly_chart(create_profit_by_category(df), use_container_width=True)
			st.plotly_chart(create_segment_distribution(df), use_container_width=True)

	elif page == "Modélisation":
		st.title("🤖 Modélisation")
		
		if st.button("Lancer l'entraînement du modèle"):
			with st.spinner("Entraînement en cours..."):
				model = SuperstoreModel()
				X, y, encoder, scaler = prepare_model_data(df, model.features)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
				
				# Entraînement et évaluation
				model.train(X_train, y_train)
				evaluation = model.evaluate(X_test, y_test)
				
				# Affichage des résultats
				st.success("Modèle entraîné avec succès!")
				st.write("Rapport de classification :")
				st.dataframe(evaluation['classification_report'])
				
				st.write("Matrice de confusion :")
				st.plotly_chart(
					create_confusion_matrix_plot(
						evaluation['confusion_matrix'],
						encoder.classes_
					)
				)

else:
	st.error("Impossible de charger les données. Veuillez vérifier le fichier source.")