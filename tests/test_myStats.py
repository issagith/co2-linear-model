import pytest
import pandas as pd
import numpy as np
import os
from linearmodel.myStats import (load_data, data_overview, columns_overview, numerical_summary, 
                                 categorical_summary, missing_values_analysis, delete_missing_values, 
                                 unique_values_analysis, correlation_analysis, outliers_count, 
                                 vehcile_outliers, variance_analysis, frequency_analysis)


@pytest.fixture
def sample_data():
    # Création d'un DataFrame mock pour les tests
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, 6, 7, 8],
        'C': ['cat', 'dog', 'cat', 'bird'],
        'D': [10.0, 10.0, 10.0, 10.0]
    })
    return data


def test_load_data(tmp_path):
    # Création d'un fichier CSV temporaire
    p = tmp_path / "test_data.csv"
    df = pd.DataFrame({'X':[1,2,3],'Y':[4,5,6]})
    df.to_csv(p, index=False)

    loaded_df = load_data(str(p))
    pd.testing.assert_frame_equal(loaded_df, df)

    # Test lorsqu'on essaie de charger un fichier inexistant
    with pytest.raises(FileNotFoundError):
        load_data("fichier_inexistant.csv")


def test_data_overview(sample_data):
    # Vérifie que data_overview renvoie les 5 premières lignes
    overview = data_overview(sample_data)
    assert len(overview) <= 5
    pd.testing.assert_frame_equal(overview, sample_data.head())


def test_columns_overview(sample_data):
    cols = columns_overview(sample_data)
    assert list(cols) == ['A', 'B', 'C', 'D']


def test_numerical_summary(sample_data):
    summary = numerical_summary(sample_data)
    # Vérifions que la description contient bien les colonnes numériques
    assert 'A' in summary.columns and 'B' in summary.columns and 'D' in summary.columns
    assert 'C' not in summary.columns  # C est catégorielle (object)


def test_categorical_summary(sample_data):
    cat_summary = categorical_summary(sample_data)
    # Vérification que la colonne 'C' (catégorielle) est présente
    assert 'C' in cat_summary.columns
    # Vérification que les colonnes numériques ne sont pas dans le résumé catégoriel
    assert 'A' not in cat_summary.columns
    assert 'B' not in cat_summary.columns


def test_missing_values_analysis(sample_data):
    missing = missing_values_analysis(sample_data)
    # On sait que 'A' contient 1 valeur manquante
    assert missing['A'] == 1
    assert missing['B'] == 0


def test_delete_missing_values(sample_data):
    cleaned_data = delete_missing_values(sample_data.copy())
    # On sait qu'une ligne contenait un NaN dans 'A', elle doit être supprimée
    assert cleaned_data.shape[0] == 3
    assert not cleaned_data.isnull().any().any()


def test_unique_values_analysis(sample_data):
    uniques = unique_values_analysis(sample_data)
    # uniques doit être un dict avec les colonnes catégorielles en clés
    assert isinstance(uniques, dict)
    assert 'C' in uniques
    # 'A', 'B', 'D' sont numériques donc pas présentes
    assert 'A' not in uniques


def test_correlation_analysis(sample_data):
    corr = correlation_analysis(sample_data)
    # Le résultat doit être une matrice de corrélation carrée sans la colonne C
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == 3  # A, B, D
    assert corr.shape[1] == 3
    assert 'C' not in corr.columns


def test_outliers_count(sample_data):
    outliers = outliers_count(sample_data)
    # outliers doit être une Series
    assert isinstance(outliers, pd.Series)
    # Dans cet exemple simple, la colonne D est constante, pas d'outliers, A et B sont simples également
    # Pas d'outliers attendus normalement
    assert (outliers == 0).all()


def test_vehcile_outliers(sample_data):
    # Cette fonction renvoie les lignes avec outliers. Sur cet échantillon simple, pas ou peu d'outliers
    outlier_rows = vehcile_outliers(sample_data)
    # On s'attend à aucune ligne car pas d'outliers stricts avec ces données
    assert outlier_rows.empty


def test_variance_analysis(sample_data):
    var = variance_analysis(sample_data)
    assert isinstance(var, pd.Series)
    # Vérifions que la variance est calculée pour les colonnes numériques
    assert all(col in var.index for col in ['A', 'B', 'D'])
    assert 'C' not in var.index


def test_frequency_analysis(sample_data):
    freq = frequency_analysis(sample_data)
    # Doit retourner un dict avec les colonnes catégorielles
    assert isinstance(freq, dict)
    assert 'C' in freq
    # Vérifions le résultat pour 'C'
    expected_counts = sample_data['C'].value_counts()
    pd.testing.assert_series_equal(freq['C'], expected_counts)

