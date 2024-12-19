import pytest
import pandas as pd
import numpy as np
from linearmodel.visualization import (
    plot_histograms, plot_boxplots, plot_histogram, plot_boxplot, plot_correlation_heatmap,
    plot_categorical_distributions, plot_categorical_distribution, plot_target_correlations,
    plot_target_correlation, plot_missing_values, plot_boxplot_by_category
)
import matplotlib.pyplot as plt

# Fixture pour désactiver l'affichage des graphiques pendant les tests
@pytest.fixture(autouse=True)
def disable_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

@pytest.fixture
def sample_data():
    """Création d'un DataFrame exemple pour les tests."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'C': ['cat', 'dog', 'cat', 'bird', 'dog'],
        'D': [10.5, np.nan, 12.3, 11.1, 10.9],
        'E': [np.nan, np.nan, 'apple', 'banana', 'apple']
    })

def test_plot_histograms(sample_data):
    plot_histograms(sample_data)
    # Vérifie simplement que la fonction ne lève pas d'erreur

def test_plot_boxplots(sample_data):
    plot_boxplots(sample_data)
    # Vérifie simplement que la fonction ne lève pas d'erreur

def test_plot_histogram(sample_data):
    # Test valide
    plot_histogram(sample_data, 'A')
    # Test avec colonne inexistante
    with pytest.raises(ValueError):
        plot_histogram(sample_data, 'Z')

def test_plot_boxplot(sample_data):
    # Test valide
    plot_boxplot(sample_data, 'A')
    # Test avec colonne inexistante
    with pytest.raises(ValueError):
        plot_boxplot(sample_data, 'Z')

def test_plot_correlation_heatmap(sample_data):
    plot_correlation_heatmap(sample_data)
    # Vérifie que la heatmap est générée avec les colonnes numériques

    # Test avec un DataFrame sans colonnes numériques
    empty_df = pd.DataFrame({'C': ['a', 'b', 'c'], 'D': ['x', 'y', 'z']})
    with pytest.raises(ValueError):
        plot_correlation_heatmap(empty_df)

def test_plot_categorical_distributions(sample_data):
    plot_categorical_distributions(sample_data)
    # Vérifie que les barplots sont générés pour les colonnes catégorielles

def test_plot_categorical_distribution(sample_data):
    # Test valide
    plot_categorical_distribution(sample_data, 'C')
    # Test avec colonne inexistante
    with pytest.raises(ValueError):
        plot_categorical_distribution(sample_data, 'Z')
    # Test avec colonne non catégorielle
    with pytest.raises(ValueError):
        plot_categorical_distribution(sample_data, 'A')

def test_plot_target_correlations(sample_data):
    sample_data['Target'] = [10, 20, 30, 40, 50]
    plot_target_correlations(sample_data, 'Target')
    # Vérifie que les scatter plots sont générés pour toutes les colonnes numériques

    # Test avec une variable cible qui n'existe pas
    with pytest.raises(ValueError):
        plot_target_correlations(sample_data, 'NonExistentTarget')

def test_plot_target_correlation(sample_data):
    sample_data['Target'] = [10, 20, 30, 40, 50]
    plot_target_correlation(sample_data, 'Target', 'A')
    # Vérifie que le scatter plot est généré pour une paire de colonnes valide

    # Test avec colonnes inexistantes
    with pytest.raises(ValueError):
        plot_target_correlation(sample_data, 'Target', 'Z')

def test_plot_missing_values(sample_data):
    plot_missing_values(sample_data)
    # Vérifie que le graphique des valeurs manquantes est généré

    # Test avec un DataFrame sans valeurs manquantes
    no_missing_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    plot_missing_values(no_missing_data)  # Aucun graphique ne doit être généré

def test_plot_boxplot_by_category(sample_data):
    sample_data['Category'] = ['x', 'y', 'x', 'y', 'x']
    plot_boxplot_by_category(sample_data, 'A', 'Category')
    # Vérifie que le boxplot est généré pour une variable numérique et une catégorielle

    # Test avec colonnes inexistantes
    with pytest.raises(ValueError):
        plot_boxplot_by_category(sample_data, 'NonExistent', 'Category')
