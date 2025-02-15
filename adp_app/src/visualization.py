import plotly.express as px
import plotly.graph_objects as go

def create_sales_by_region(df):
    """Crée un graphique des ventes par région"""
    fig = px.bar(
        df.groupby('Region')['Sales'].sum().reset_index(),
        x='Region',
        y='Sales',
        title="Total des Ventes par Région",
        template="plotly_white"
    )
    return fig

def create_profit_by_category(df):
    """Crée un graphique des profits par catégorie"""
    fig = px.bar(
        df.groupby('Category')['Profit'].sum().reset_index(),
        x='Category',
        y='Profit',
        title="Profit Total par Catégorie",
        template="plotly_white"
    )
    return fig

def create_monthly_sales_trend(df):
    """Crée un graphique de tendance des ventes mensuelles"""
    df['Order Month'] = pd.to_datetime(df['Order Date']).dt.to_period('M')
    monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()
    monthly_sales['Order Month'] = monthly_sales['Order Month'].astype(str)
    
    fig = px.line(
        monthly_sales,
        x='Order Month',
        y='Sales',
        title="Tendance des Ventes Mensuelles",
        template="plotly_white"
    )
    return fig

def create_segment_distribution(df):
    """Crée un graphique de distribution des segments"""
    fig = px.pie(
        df,
        names='Segment',
        title="Distribution des Segments Clients",
        template="plotly_white"
    )
    return fig