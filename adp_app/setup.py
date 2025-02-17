from setuptools import setup, find_packages

setup(
    name="superstore-analytics",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.31.1',
        'pandas>=2.1.4',
        'numpy>=1.24.3',
        'plotly>=5.18.0',
        'scikit-learn>=1.3.2',
        'seaborn>=0.13.0',
        'openpyxl>=3.1.2',
        'pathlib>=1.0.1',
        'scipy>=1.11.3',
        'matplotlib>=3.7.1'
    ],
)